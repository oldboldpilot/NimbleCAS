// NimbleCAS splitting-field construction over a number field on the UNBOUNDED rationals:
// Trager's norm-based factorization of BigAlgebraicPoly over a BigNumberField, adjunction of
// a single root of an irreducible extension polynomial via a primitive-element search, and
// iterated adjunction to build the splitting field of a batch of Q-irreducible polynomials
// (§7 algebra substrate; the bignum mirror of nimblecas.splitfield, built on
// nimblecas.bigalgnum / nimblecas.bigalgpoly).
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision analogue of nimblecas.splitfield. Where that module carries
// int64 Rational / RationalPoly coefficients and surfaces MathError::overflow the moment an
// intermediate numerator or denominator saturates, this one carries BigRational /
// BigRationalPoly coefficients and has NO such ceiling. That is the whole point of this tier:
// the int64 splitfield OVERFLOWS while building the degree-6 splitting field of x^3 - 2 (its
// Trager norms and primitive-element multiplication matrices dwarf the ~9.22e18 int64
// envelope), whereas here every intermediate is the exact element of Q it mathematically is,
// so the construction runs to completion. MathError::overflow can NEVER occur on this tier.
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
// iff that matrix's minimal polynomial (via nimblecas.bigfrobenius) has degree exactly n.
//
// HONESTY DIVERGENCE FROM the int64 splitfield (documented, load-bearing). In the int64 tier
// the degree-n minimal polynomial h' is automatically irreducible when M0 truly is a field,
// and NumberField::create(h') honestly REJECTS a reducible h' (the signal that g was not
// actually irreducible over L, so M0 is a product of smaller fields). But BigNumberField at
// this tier has NO polynomial-factoring layer: BigNumberField::create does NOT prove
// irreducibility -- it only normalises to monic. So the primitivity criterion "minimal-
// polynomial degree == n" is INSUFFICIENT on its own here. To preserve the exact honesty
// invariant, this module additionally verifies h' is Q-IRREDUCIBLE via
// bigfactor::factor_over_Q(h'), confirming a single irreducible factor of multiplicity 1 and
// degree n; a nontrivial factorization means gamma' is NOT primitive (g was not actually
// irreducible over L), so the bounded search continues, and an exhausted search returns
// MathError::not_implemented -- exactly the honest signal the int64 path emits. Once h' is
// proven irreducible, M = Q[z]/(h') is built, then alpha and beta are recovered as
// Q-polynomials in gamma' by solving the (invertible, since gamma' is primitive) linear
// system expressing the powers gamma'^0..gamma'^(n-1) against the targets "alpha" and "beta"
// in the same coordinates. Before returning, both g(root) == 0 and L.modulus()(old_generator)
// == 0 are independently re-verified in M (Rule 32): an embedding bug is caught at the source.
//
// SPLITTING_FIELD. Input factors are sorted into a canonical order (degree ascending, then
// coefficient-lexicographic) purely to make the field-growth sequence deterministic; results
// are reported back in the CALLER's original order. The starting field is always the trivial
// degree-1 field Q[t]/(t) (representing Q itself), grown one adjoined root at a time. The
// LOAD-BEARING design choice is INCREMENTAL ROOT-DIVISION: rather than re-factor the FULL
// input p_i over each enlarged field (which for x^3 - 2 over its degree-6 splitting field
// would demand a Trager norm of degree [L:Q]*deg = 6*3 = 18, far beyond factor_over_Q's
// Kronecker budget), the driver keeps, per input, only the not-yet-split COFACTOR: each pass
// pulls out every linear factor cheaply, notes the first non-linear irreducible piece as the
// next adjunction target, then -- after adjoin_root extends the field -- re-embeds every
// cofactor and found root forward and DIVIDES the freshly adjoined root out of its cofactor.
// Every polynomial ever handed to factor_over_field is therefore the reduced cofactor, whose
// Trager norm stays low (<= 6 for x^3 - 2), so the construction the int64 tier cannot even
// reach by overflow runs to completion here. The loop stops when every cofactor is fully
// split into linear factors; the completeness invariant (each input contributes exactly deg_i
// pairwise-distinct roots) is re-verified (Rule 32) before the roots are reported.
//
// HONESTY (Rule 32). Nothing throws; nothing returns a plausible-but-wrong field or root.
// Because BigRational / BigRationalPoly arithmetic only combines magnitudes, it CANNOT
// overflow -- MathError::overflow can NEVER arise on this tier, which is exactly the int64
// splitfield's overflow boundary that this module removes. [L:Q] exceeding max_degree at any
// point -- the initial field, or a subsequent adjunction -- is MathError::not_implemented (a
// quartic with Galois group S4 needs degree 24; this module refuses rather than guess).
// Exhausting the shift bound in Trager's algorithm (no squarefree norm found) or the
// primitive-element search bound in adjoin_root is MathError::not_implemented, and
// factor_over_Q's own internal budget propagates as not_implemented. Field-mismatch,
// degree < 2 in adjoin_root, and any embedding-verification failure are domain_error. The
// practical envelope, in line with factor_over_Q and BigNumberField, is roughly degree <= 6
// with small coefficients; beyond that, an honest not_implemented is preferred over a wrong
// answer -- but within it, this tier completes cases (like x^3 - 2's degree-6 splitting
// field) that the int64 tier cannot, because its intermediate arithmetic overflows.

module;
#include <cassert>

export module nimblecas.bigsplitfield;

import std;
import nimblecas.core;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
import nimblecas.bigmatrix;
import nimblecas.bigalgnum;
import nimblecas.bigalgpoly;
import nimblecas.bigfactor;
import nimblecas.bigresultant;
import nimblecas.bigfrobenius;

export namespace nimblecas {

// Factor a SQUAREFREE f in L[x] into its monic L-irreducible factors, via Trager's norm
// algorithm. f must live in `l` (else domain_error); the zero polynomial is domain_error.
// A nonzero constant f yields an empty list (no non-unit factors); a linear f yields itself
// (monic). Fails with MathError::not_implemented when no shift in the bounded search
// |s| <= 2*[L:Q]*deg(f) yields a squarefree norm, or when factor_over_Q(N) exceeds its own
// budget. The exact bignum arithmetic never overflows. The reconstruction
// f == lc(f) * prod(factors) is verified before returning (domain_error on mismatch).
[[nodiscard]] auto factor_over_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicPoly>>;

// The roots of f that lie in L, read off the degree-1 factors of factor_over_field(l, f)
// as lambda = -c0/c1. Order matches factor_over_field's (unspecified but deterministic)
// factor order. Errors propagate from factor_over_field.
[[nodiscard]] auto roots_in_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicNumber>>;

// A simple extension M >= L containing a root of g, plus the images in M of L's generator
// (`old_generator`, needed to re-embed any other element of L into M) and of the adjoined
// root itself (`root`).
struct BigAdjoinedRoot {
    BigNumberField field;
    BigAlgebraicNumber old_generator;
    BigAlgebraicNumber root;
};

// Adjoin a root of g (irreducible over L, deg g = e >= 2) to L, building M = Q(gamma') as a
// SIMPLE extension via a primitive-element search gamma' = beta + c*alpha over
// c = 1, -1, 2, -2, .... Fails with domain_error when g does not live in `l` or has degree
// < 2; with MathError::not_implemented when no primitive element is found within the bounded
// search (this is also the honest signal if g was not actually irreducible over L, since this
// tier proves primitivity by verifying h' is Q-irreducible via factor_over_Q -- see the module
// header), or when factor_over_Q exceeds its own budget; with domain_error if the mandatory
// post-construction verification (g(root) == 0 and L.modulus()(old_generator) == 0, both
// checked in M) fails. The exact bignum arithmetic never overflows.
[[nodiscard]] auto adjoin_root(const BigNumberField& l, const BigAlgebraicPoly& g)
    -> Result<BigAdjoinedRoot>;

// The splitting field of a batch of Q-irreducible polynomials, together with each
// polynomial's full set of roots in that field (in the SAME order as `irreducibles`).
struct BigSplittingField {
    BigNumberField field;
    std::vector<std::pair<BigRationalPoly, std::vector<BigAlgebraicNumber>>> roots;
};

// Build the splitting field of `irreducibles` (each assumed Q-irreducible; a non-positive
// degree entry is domain_error), refusing to build a field of degree exceeding
// `max_degree` at any point (the initial field or any subsequent adjunction) with
// MathError::not_implemented -- the honest boundary, since e.g. a quartic with Galois
// group S4 needs degree 24. Errors from factor_over_field / adjoin_root propagate
// (not_implemented / domain_error; overflow is impossible on this tier). A defensive
// completeness guard (every factor contributes exactly deg_i pairwise-distinct roots) is
// re-verified before returning; a violation is domain_error (an embedding bug, not a
// mathematical possibility for a genuinely Q-irreducible, hence separable, input).
[[nodiscard]] auto splitting_field(std::span<const BigRationalPoly> irreducibles,
                                    std::int64_t max_degree) -> Result<BigSplittingField>;

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

// Exact a < b for BigRationals. Unlike the int64 tier -- where rational_less cross-multiplies
// through __int128 to dodge an int64 overflow guard -- BigRational's own operator<=> is exact
// and unbounded, so the comparison is used directly. Used only to establish a deterministic
// sort order, never as a computed mathematical result.
[[nodiscard]] auto bigrational_less(const BigRational& a, const BigRational& b) -> bool {
    return a < b;
}

// Canonical order over BigRationalPoly: degree ascending, then coefficient-lexicographic
// (constant term first). Used only to make the outer field-growth algorithm deterministic.
[[nodiscard]] auto poly_less(const BigRationalPoly& a, const BigRationalPoly& b) -> bool {
    if (a.degree() != b.degree()) {
        return a.degree() < b.degree();
    }
    for (std::int64_t i = 0; i <= a.degree(); ++i) {
        const BigRational ca = a.coefficient(static_cast<std::size_t>(i));
        const BigRational cb = b.coefficient(static_cast<std::size_t>(i));
        if (!(ca == cb)) {
            return bigrational_less(ca, cb);
        }
    }
    return false;
}

// The deterministic rational sample sequence 0, 1, -1, 2, -2, ..., `count` values.
[[nodiscard]] auto sample_nodes(std::size_t count) -> std::vector<BigRational> {
    std::vector<BigRational> nodes;
    nodes.reserve(count);
    if (count == 0) {
        return nodes;
    }
    nodes.push_back(BigRational::from_int(0));
    std::int64_t k = 1;
    while (nodes.size() < count) {
        nodes.push_back(BigRational::from_int(k));
        if (nodes.size() < count) {
            nodes.push_back(BigRational::from_int(-k));
        }
        ++k;
    }
    return nodes;
}

// Lagrange-interpolate the unique polynomial of degree < nodes.size() through the given
// (node, value) pairs (nodes assumed distinct). The BigRational / BigRationalPoly add,
// subtract, multiply and scale used here are INFALLIBLE (arbitrary precision cannot
// overflow), so unlike the int64 tier the only fallible step is the reciprocal of the
// node-difference product, which is non-zero for distinct nodes.
[[nodiscard]] auto lagrange_interpolate(std::span<const BigRational> nodes,
                                        std::span<const BigRational> values)
    -> Result<BigRationalPoly> {
    const std::size_t m = nodes.size();
    BigRationalPoly acc;  // zero
    for (std::size_t i = 0; i < m; ++i) {
        if (values[i].is_zero()) {
            continue;
        }
        BigRationalPoly num = BigRationalPoly::constant(BigRational::from_int(1));
        BigRational denom = BigRational::from_int(1);
        for (std::size_t j = 0; j < m; ++j) {
            if (j == i) {
                continue;
            }
            const BigRational neg_nj = nodes[j].negate();  // infallible over BigRational
            const BigRationalPoly lin =
                BigRationalPoly::from_coeffs(std::vector<BigRational>{neg_nj, BigRational::from_int(1)});
            num = num.multiply(lin);                        // infallible
            const BigRational diff = nodes[i].subtract(nodes[j]);  // infallible
            denom = denom.multiply(diff);                   // infallible
        }
        auto inv_denom = BigRational::from_int(1).divide(denom);  // denom != 0: distinct nodes
        if (!inv_denom) {
            return make_error<BigRationalPoly>(inv_denom.error());
        }
        const BigRational coeff = values[i].multiply(*inv_denom);  // infallible
        const BigRationalPoly term = num.scale(coeff);             // infallible
        acc = acc.add(term);                                       // infallible
    }
    return acc;
}

// --- BigAlgebraicPoly helpers -------------------------------------------------

// f(x - a): polynomial composition with the linear shift (x - a), via Horner's scheme.
// a and f must live in the same field.
[[nodiscard]] auto shift_by(const BigAlgebraicPoly& f, const BigAlgebraicNumber& a)
    -> Result<BigAlgebraicPoly> {
    const BigNumberField field = f.field();
    auto neg_a = a.negate();
    if (!neg_a) {
        return make_error<BigAlgebraicPoly>(neg_a.error());
    }
    auto linear_res =
        BigAlgebraicPoly::from_coeffs(field, std::vector<BigAlgebraicNumber>{*neg_a, field.one()});
    if (!linear_res) {
        return make_error<BigAlgebraicPoly>(linear_res.error());
    }
    const BigAlgebraicPoly linear = std::move(*linear_res);
    BigAlgebraicPoly acc = BigAlgebraicPoly::zero(field);
    for (std::int64_t i = f.degree(); i >= 0; --i) {
        auto prod = acc.multiply(linear);
        if (!prod) {
            return make_error<BigAlgebraicPoly>(prod.error());
        }
        auto ci_res = BigAlgebraicPoly::from_coeffs(
            field, std::vector<BigAlgebraicNumber>{f.coefficient(static_cast<std::size_t>(i))});
        if (!ci_res) {
            return make_error<BigAlgebraicPoly>(ci_res.error());
        }
        auto sum = prod->add(*ci_res);
        if (!sum) {
            return make_error<BigAlgebraicPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

// g(x0, t): the BigAlgebraicPoly g over L = Q[t]/(h(t)) evaluated at x = x0 (a plain
// rational), WITHOUT reducing the field generator further -- i.e. the Q[t] polynomial
// sum_i g_i(t) * x0^i where g_i(t) is the canonical residue of g's i-th coefficient. This
// is exactly the object whose resultant against h gives Norm_{L/Q}(g(x0)). The residue
// scaling and accumulation are infallible over BigRational, so the Result is threaded only
// for signature parity.
[[nodiscard]] auto eval_coeffs_at_x(const BigAlgebraicPoly& g, const BigRational& x0)
    -> Result<BigRationalPoly> {
    BigRationalPoly acc;  // zero
    BigRational power = BigRational::from_int(1);
    const std::int64_t deg = g.degree();
    for (std::int64_t i = 0; i <= deg; ++i) {
        const BigAlgebraicNumber ci = g.coefficient(static_cast<std::size_t>(i));
        const BigRationalPoly term = ci.value().scale(power);  // infallible
        acc = acc.add(term);                                   // infallible
        if (i < deg) {
            power = power.multiply(x0);                        // infallible
        }
    }
    return acc;
}

// p reduced modulo the (nonzero) divisor g.
[[nodiscard]] auto reduce_mod(const BigAlgebraicPoly& p, const BigAlgebraicPoly& g)
    -> Result<BigAlgebraicPoly> {
    auto dm = p.divide(g);
    if (!dm) {
        return make_error<BigAlgebraicPoly>(dm.error());
    }
    return std::move(dm->remainder);
}

// Re-embed a field element x (living in some field whose generator maps to
// `image_of_generator` in `new_field`) into `new_field`, by lifting x's canonical residue
// polynomial and evaluating it at the generator's image. This is the ring homomorphism
// induced by generator |-> image_of_generator.
[[nodiscard]] auto reembed(const BigAlgebraicNumber& x, const BigNumberField& new_field,
                           const BigAlgebraicNumber& image_of_generator)
    -> Result<BigAlgebraicNumber> {
    const BigRationalPoly rep = x.value();
    const BigAlgebraicPoly lifted = BigAlgebraicPoly::embed(new_field, rep);
    return lifted.evaluate(image_of_generator);
}

// Coordinate map for adjoin_root's Q-algebra M0 = L[y]/(g): an M0 element, represented as
// a BigAlgebraicPoly p over L of degree < e (coefficient j is p's L-coefficient of y^j), is
// mapped to its n = d*e coordinates on the basis {alpha^i * beta^j}, index i*e + j.
[[nodiscard]] auto to_vector(const BigAlgebraicPoly& p, std::int64_t d, std::int64_t e)
    -> std::vector<BigRational> {
    std::vector<BigRational> vec(static_cast<std::size_t>(d) * static_cast<std::size_t>(e));
    for (std::int64_t j = 0; j < e; ++j) {
        const BigAlgebraicNumber cj = p.coefficient(static_cast<std::size_t>(j));
        const BigRationalPoly& val = cj.value();
        for (std::int64_t i = 0; i < d; ++i) {
            vec[static_cast<std::size_t>(i * e + j)] = val.coefficient(static_cast<std::size_t>(i));
        }
    }
    return vec;
}

// The constant polynomial 1 over `field` -- the "fully split" cofactor sentinel. Built via
// embed (infallible) rather than from_coeffs, so no Result has to be threaded.
[[nodiscard]] auto poly_one(const BigNumberField& field) -> BigAlgebraicPoly {
    return BigAlgebraicPoly::embed(field, BigRationalPoly::constant(BigRational::from_int(1)));
}

// Re-embed every coefficient of a BigAlgebraicPoly over L into new_field, under the ring
// homomorphism induced by L's generator |-> image_of_generator. Used by splitting_field to
// carry a cofactor forward when the field grows.
[[nodiscard]] auto reembed_poly(const BigAlgebraicPoly& p, const BigNumberField& new_field,
                                const BigAlgebraicNumber& image_of_generator)
    -> Result<BigAlgebraicPoly> {
    if (p.is_zero()) {
        return BigAlgebraicPoly::zero(new_field);
    }
    std::vector<BigAlgebraicNumber> coeffs;
    coeffs.reserve(static_cast<std::size_t>(p.degree()) + 1);
    for (std::int64_t i = 0; i <= p.degree(); ++i) {
        auto c = reembed(p.coefficient(static_cast<std::size_t>(i)), new_field, image_of_generator);
        if (!c) {
            return make_error<BigAlgebraicPoly>(c.error());
        }
        coeffs.push_back(std::move(*c));
    }
    return BigAlgebraicPoly::from_coeffs(new_field, std::move(coeffs));
}

// The root of a linear factor l = c1*x + c0 over its field: -c0/c1 (c1 != 0 for a genuine
// degree-1 polynomial).
[[nodiscard]] auto linear_root(const BigAlgebraicPoly& l) -> Result<BigAlgebraicNumber> {
    const BigAlgebraicNumber c0 = l.coefficient(0);
    const BigAlgebraicNumber c1 = l.coefficient(1);
    auto neg = c0.negate();
    if (!neg) {
        return make_error<BigAlgebraicNumber>(neg.error());
    }
    return neg->divide(c1);
}

// poly / (x - root), which must divide exactly over poly's field. A nonzero remainder is an
// embedding bug, reported as domain_error rather than a fabricated quotient (Rule 32).
[[nodiscard]] auto divide_out_root(const BigAlgebraicPoly& poly, const BigAlgebraicNumber& root,
                                   const BigNumberField& field) -> Result<BigAlgebraicPoly> {
    auto neg = root.negate();
    if (!neg) {
        return make_error<BigAlgebraicPoly>(neg.error());
    }
    auto lin = BigAlgebraicPoly::from_coeffs(field, std::vector<BigAlgebraicNumber>{*neg, field.one()});
    if (!lin) {
        return make_error<BigAlgebraicPoly>(lin.error());
    }
    auto dm = poly.divide(*lin);
    if (!dm) {
        return make_error<BigAlgebraicPoly>(dm.error());
    }
    if (!dm->remainder.is_zero()) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    return std::move(dm->quotient);
}

}  // namespace

// --- factor_over_field --------------------------------------------------------

auto factor_over_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicPoly>> {
    using Out = std::vector<BigAlgebraicPoly>;
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
    const BigAlgebraicNumber gamma = std::move(*gamma_res);
    const BigRationalPoly h = l.modulus();

    // deg N(x) = [L:Q] * deg f. This is a pure int64 DEGREE product (not coefficient
    // arithmetic, which cannot overflow on this tier); a degree beyond int64 is physically
    // unreachable, and if it ever were, it is something this module cannot handle -- reported
    // as not_implemented rather than a fabricated overflow the bignum arithmetic never emits.
    auto deg_n_opt = checked_mul_i64(d, deg_f);
    if (!deg_n_opt) {
        return make_error<Out>(MathError::not_implemented);
    }
    const std::int64_t deg_n = *deg_n_opt;
    const std::int64_t bound = checked_mul_i64(2, deg_n).value_or(std::numeric_limits<std::int64_t>::max());

    for (std::int64_t k = 0;; ++k) {
        std::int64_t s = 0;
        if (k > 0) {
            s = (k % 2 == 1) ? (k + 1) / 2 : -(k / 2);
        }
        if (s > bound || s < -bound) {
            return make_error<Out>(MathError::not_implemented);
        }

        const BigAlgebraicNumber s_elem = l.from_bigrational(BigRational::from_int(s));
        auto shift_amt = gamma.multiply(s_elem);
        if (!shift_amt) {
            return make_error<Out>(shift_amt.error());
        }
        auto g_res = shift_by(f, *shift_amt);  // g(x) = f(x - s*gamma)
        if (!g_res) {
            return make_error<Out>(g_res.error());
        }
        const BigAlgebraicPoly g = std::move(*g_res);

        const std::vector<BigRational> nodes = sample_nodes(static_cast<std::size_t>(deg_n) + 1);
        std::vector<BigRational> values;
        values.reserve(nodes.size());
        bool arith_error = false;
        MathError err = MathError::domain_error;
        for (const BigRational& x0 : nodes) {
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
        const BigRationalPoly N = std::move(*n_res);

        // f's (hence g's) leading coefficient is a nonzero field element, so its norm is
        // nonzero and deg N must be exactly deg_n; a mismatch is an interpolation bug, not
        // a mathematical possibility, and is reported honestly rather than risked further.
        assert(N.degree() == deg_n && "Trager norm degree must equal [L:Q] * deg f");
        if (N.degree() != deg_n) {
            return make_error<Out>(MathError::domain_error);
        }

        const BigRationalPoly n_prime = N.derivative();  // infallible over BigRational
        auto sqfree_gcd = N.gcd(n_prime);
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
            const BigAlgebraicPoly ni_l = BigAlgebraicPoly::embed(l, ni);
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
        auto product_res = BigAlgebraicPoly::from_coeffs(l, std::vector<BigAlgebraicNumber>{l.one()});
        if (!product_res) {
            return make_error<Out>(product_res.error());
        }
        BigAlgebraicPoly product = std::move(*product_res);
        for (const BigAlgebraicPoly& fac : result) {
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

auto roots_in_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicNumber>> {
    using Out = std::vector<BigAlgebraicNumber>;
    auto factors = factor_over_field(l, f);
    if (!factors) {
        return make_error<Out>(factors.error());
    }
    Out roots;
    for (const BigAlgebraicPoly& fac : *factors) {
        if (fac.degree() != 1) {
            continue;
        }
        const BigAlgebraicNumber c0 = fac.coefficient(0);
        const BigAlgebraicNumber c1 = fac.coefficient(1);
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

auto adjoin_root(const BigNumberField& l, const BigAlgebraicPoly& g) -> Result<BigAdjoinedRoot> {
    if (!g.field().is_same(l)) {
        return make_error<BigAdjoinedRoot>(MathError::domain_error);
    }
    const std::int64_t e = g.degree();
    if (e < 2) {
        return make_error<BigAdjoinedRoot>(MathError::domain_error);
    }
    const std::int64_t d = l.degree();
    // n = d * e is a pure int64 DEGREE product; a degree beyond int64 is physically
    // unreachable (coefficient arithmetic cannot overflow on this tier), so if it ever were,
    // it is beyond what this module can build -- reported as not_implemented, never a
    // fabricated overflow.
    auto n_opt = checked_mul_i64(d, e);
    if (!n_opt) {
        return make_error<BigAdjoinedRoot>(MathError::not_implemented);
    }
    const std::int64_t n = *n_opt;
    const std::size_t un = static_cast<std::size_t>(n);

    auto alpha_res = l.generator();
    if (!alpha_res) {
        return make_error<BigAdjoinedRoot>(alpha_res.error());
    }
    const BigAlgebraicNumber alpha = std::move(*alpha_res);

    // target_alpha: the M0 = L[y]/(g) element equal to alpha itself (y^0 coefficient).
    auto target_alpha_res = BigAlgebraicPoly::from_coeffs(l, std::vector<BigAlgebraicNumber>{alpha});
    if (!target_alpha_res) {
        return make_error<BigAdjoinedRoot>(target_alpha_res.error());
    }
    // target_beta: the M0 element equal to y (= beta) itself.
    auto target_beta_res =
        BigAlgebraicPoly::from_coeffs(l, std::vector<BigAlgebraicNumber>{l.zero(), l.one()});
    if (!target_beta_res) {
        return make_error<BigAdjoinedRoot>(target_beta_res.error());
    }
    const std::vector<BigRational> target_alpha_vec = to_vector(*target_alpha_res, d, e);
    const std::vector<BigRational> target_beta_vec = to_vector(*target_beta_res, d, e);

    constexpr std::int64_t kMaxPrimitiveSearch = 64;
    for (std::int64_t k = 1; k <= kMaxPrimitiveSearch; ++k) {
        const std::int64_t c = (k % 2 == 1) ? (k + 1) / 2 : -(k / 2);  // 1, -1, 2, -2, ...

        const BigAlgebraicNumber c_elem = l.from_bigrational(BigRational::from_int(c));
        auto c_alpha = alpha.multiply(c_elem);
        if (!c_alpha) {
            return make_error<BigAdjoinedRoot>(c_alpha.error());
        }
        auto gamma_prime_res =
            BigAlgebraicPoly::from_coeffs(l, std::vector<BigAlgebraicNumber>{*c_alpha, l.one()});
        if (!gamma_prime_res) {
            return make_error<BigAdjoinedRoot>(gamma_prime_res.error());
        }
        const BigAlgebraicPoly gamma_prime = std::move(*gamma_prime_res);

        // The n x n rational matrix of multiplication-by-gamma' on the Q-basis
        // { alpha^i * beta^j : 0 <= i < d, 0 <= j < e }, column index i*e + j.
        std::vector<std::vector<BigRational>> rows(un, std::vector<BigRational>(un));
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
                std::vector<BigAlgebraicNumber> basis_coeffs(static_cast<std::size_t>(j) + 1, l.zero());
                basis_coeffs[static_cast<std::size_t>(j)] = *a_i;
                auto basis_elem = BigAlgebraicPoly::from_coeffs(l, std::move(basis_coeffs));
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
                const std::vector<BigRational> col = to_vector(*reduced, d, e);
                const std::size_t col_idx = static_cast<std::size_t>(i * e + j);
                for (std::size_t row = 0; row < un; ++row) {
                    rows[row][col_idx] = col[row];
                }
            }
        }
        if (build_error) {
            return make_error<BigAdjoinedRoot>(berr);
        }
        auto mmat = BigMatrix::from_rows(std::move(rows));
        if (!mmat) {
            return make_error<BigAdjoinedRoot>(mmat.error());
        }

        auto minpoly = minimal_polynomial(*mmat);
        if (!minpoly) {
            return make_error<BigAdjoinedRoot>(minpoly.error());
        }
        if (minpoly->degree() != n) {
            continue;  // gamma' not primitive: try the next c
        }

        // HONESTY DIVERGENCE from the int64 tier (see module header). NumberField::create
        // would reject a reducible minimal polynomial there; BigNumberField::create does NOT
        // prove irreducibility (no bignum factoring layer). So a degree-n minimal polynomial
        // is NOT sufficient evidence of primitivity here -- if g were not actually irreducible
        // over L, M0 is a product of smaller fields and any degree-n minpoly found there is
        // REDUCIBLE. We prove h' is Q-irreducible via factor_over_Q: a single irreducible
        // factor of multiplicity 1 and degree n. A nontrivial factorization means gamma' is
        // not primitive, so the bounded search continues (an exhausted search returns
        // not_implemented below -- the honest signal the int64 path emits).
        auto minpoly_factors = factor_over_Q(*minpoly);
        if (!minpoly_factors) {
            return make_error<BigAdjoinedRoot>(minpoly_factors.error());  // factor budget
        }
        if (minpoly_factors->size() != 1 || (*minpoly_factors)[0].second != 1 ||
            (*minpoly_factors)[0].first.degree() != n) {
            continue;  // h' reducible => gamma' not primitive: try the next c
        }

        auto field_res = BigNumberField::create(*minpoly);
        if (!field_res) {
            return make_error<BigAdjoinedRoot>(field_res.error());
        }
        const BigNumberField m_field = std::move(*field_res);

        // Powers gamma'^0 .. gamma'^(n-1) as M0 elements, each an n-dim Q-vector: the
        // columns of the change-of-basis matrix V.
        std::vector<std::vector<BigRational>> vrows(un, std::vector<BigRational>(un));
        auto one_res = BigAlgebraicPoly::from_coeffs(l, std::vector<BigAlgebraicNumber>{l.one()});
        if (!one_res) {
            return make_error<BigAdjoinedRoot>(one_res.error());
        }
        BigAlgebraicPoly power = std::move(*one_res);
        bool pow_error = false;
        MathError perr = MathError::domain_error;
        for (std::int64_t col = 0; col < n; ++col) {
            const std::vector<BigRational> pv = to_vector(power, d, e);
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
            return make_error<BigAdjoinedRoot>(perr);
        }
        auto vmat = BigMatrix::from_rows(std::move(vrows));
        if (!vmat) {
            return make_error<BigAdjoinedRoot>(vmat.error());
        }

        std::vector<std::vector<BigRational>> brows(un, std::vector<BigRational>(2));
        for (std::size_t row = 0; row < un; ++row) {
            brows[row][0] = target_alpha_vec[row];
            brows[row][1] = target_beta_vec[row];
        }
        auto bmat = BigMatrix::from_rows(std::move(brows));
        if (!bmat) {
            return make_error<BigAdjoinedRoot>(bmat.error());
        }
        auto xmat = vmat->solve(*bmat);  // V invertible: gamma' is primitive (checked above)
        if (!xmat) {
            return make_error<BigAdjoinedRoot>(xmat.error());
        }

        std::vector<BigRational> alpha_coeffs(un);
        std::vector<BigRational> beta_coeffs(un);
        for (std::size_t row = 0; row < un; ++row) {
            alpha_coeffs[row] = xmat->at(row, 0);
            beta_coeffs[row] = xmat->at(row, 1);
        }
        const BigRationalPoly p_alpha = BigRationalPoly::from_coeffs(std::move(alpha_coeffs));
        const BigRationalPoly p_beta = BigRationalPoly::from_coeffs(std::move(beta_coeffs));

        auto alpha_in_m = m_field.from_poly(p_alpha);
        if (!alpha_in_m) {
            return make_error<BigAdjoinedRoot>(alpha_in_m.error());
        }
        auto root_in_m = m_field.from_poly(p_beta);
        if (!root_in_m) {
            return make_error<BigAdjoinedRoot>(root_in_m.error());
        }

        // VERIFY before returning (mandatory, Rule 32): g's image over M vanishes at
        // `root`, and L's modulus vanishes at the image of L's generator in M.
        std::vector<BigAlgebraicNumber> g_in_m_coeffs;
        g_in_m_coeffs.reserve(static_cast<std::size_t>(g.degree()) + 1);
        bool embed_error = false;
        MathError eerr = MathError::domain_error;
        for (std::int64_t j = 0; j <= g.degree(); ++j) {
            const BigAlgebraicNumber cj = g.coefficient(static_cast<std::size_t>(j));
            auto img = reembed(cj, m_field, *alpha_in_m);
            if (!img) {
                embed_error = true;
                eerr = img.error();
                break;
            }
            g_in_m_coeffs.push_back(std::move(*img));
        }
        if (embed_error) {
            return make_error<BigAdjoinedRoot>(eerr);
        }
        auto g_in_m = BigAlgebraicPoly::from_coeffs(m_field, std::move(g_in_m_coeffs));
        if (!g_in_m) {
            return make_error<BigAdjoinedRoot>(g_in_m.error());
        }
        auto g_at_root = g_in_m->evaluate(*root_in_m);
        if (!g_at_root) {
            return make_error<BigAdjoinedRoot>(g_at_root.error());
        }
        if (!g_at_root->is_zero()) {
            return make_error<BigAdjoinedRoot>(MathError::domain_error);
        }

        const BigAlgebraicPoly h_in_m = BigAlgebraicPoly::embed(m_field, l.modulus());
        auto h_at_alpha = h_in_m.evaluate(*alpha_in_m);
        if (!h_at_alpha) {
            return make_error<BigAdjoinedRoot>(h_at_alpha.error());
        }
        if (!h_at_alpha->is_zero()) {
            return make_error<BigAdjoinedRoot>(MathError::domain_error);
        }

        return BigAdjoinedRoot{.field = m_field, .old_generator = std::move(*alpha_in_m),
                                .root = std::move(*root_in_m)};
    }
    return make_error<BigAdjoinedRoot>(MathError::not_implemented);
}

// --- splitting_field ----------------------------------------------------------------

auto splitting_field(std::span<const BigRationalPoly> irreducibles, std::int64_t max_degree)
    -> Result<BigSplittingField> {
    const std::size_t m = irreducibles.size();
    for (const BigRationalPoly& p : irreducibles) {
        if (p.degree() < 1) {
            return make_error<BigSplittingField>(MathError::domain_error);
        }
    }

    // Canonical order (degree ascending, then coefficient-lexicographic): a deterministic
    // adjunction order only. Results are reported back in the CALLER's original order (below).
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
        return poly_less(irreducibles[a], irreducibles[b]);
    });

    // Start at the trivial degree-1 field Q[t]/(t) = Q, and grow it one adjoined root at a
    // time. NOTE the load-bearing design choice: rather than re-factor the FULL input p_i
    // over each enlarged field (which, for x^3 - 2 over its degree-6 splitting field, would
    // demand a Trager norm of degree [L:Q]*deg = 6*3 = 18 -- far beyond factor_over_Q's
    // Kronecker budget), we keep, per input, only the not-yet-split COFACTOR and divide out
    // each root the moment it is found. Every polynomial ever handed to factor_over_field is
    // therefore the reduced cofactor, whose Trager norm stays low (<= 6 for x^3 - 2), so the
    // construction the int64 tier cannot even reach by overflow runs to completion here.
    auto trivial_res = BigNumberField::create(BigRationalPoly::monomial(BigRational::from_int(1), 1));
    if (!trivial_res) {
        return make_error<BigSplittingField>(trivial_res.error());
    }
    BigNumberField field = std::move(*trivial_res);

    std::vector<BigAlgebraicPoly> remaining;  // cofactor of each p_i over the CURRENT field
    remaining.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        remaining.push_back(BigAlgebraicPoly::embed(field, irreducibles[i]));
    }
    std::vector<std::vector<BigAlgebraicNumber>> found(m);  // roots so far, re-embedded forward

    // Bounded outer loop: each adjunction strictly raises [L:Q] (deg target >= 2) toward the
    // max_degree ceiling, so the count is bounded. This cap is a defensive circuit breaker,
    // not the honesty gate itself -- that gate is the max_degree check below.
    const std::int64_t iteration_cap =
        std::max<std::int64_t>(max_degree, 1) + static_cast<std::int64_t>(m) + 8;

    for (std::int64_t iter = 0;; ++iter) {
        if (iter > iteration_cap) {
            return make_error<BigSplittingField>(MathError::not_implemented);
        }

        // Sweep every cofactor over the current field: pull out all linear factors (cheap --
        // no adjunction), and note the first non-linear irreducible piece as the next
        // adjunction target.
        std::optional<BigAlgebraicPoly> target_piece;
        std::size_t target_i = 0;
        for (std::size_t idx : order) {
            if (remaining[idx].degree() < 1) {
                continue;  // already fully split
            }
            if (remaining[idx].degree() == 1) {
                auto r = linear_root(remaining[idx]);
                if (!r) {
                    return make_error<BigSplittingField>(r.error());
                }
                found[idx].push_back(std::move(*r));
                remaining[idx] = poly_one(field);
                continue;
            }
            auto facs = factor_over_field(field, remaining[idx]);
            if (!facs) {
                return make_error<BigSplittingField>(facs.error());
            }
            BigAlgebraicPoly nonlinear_product = poly_one(field);
            for (const BigAlgebraicPoly& piece : *facs) {
                if (piece.degree() == 1) {
                    auto r = linear_root(piece);
                    if (!r) {
                        return make_error<BigSplittingField>(r.error());
                    }
                    found[idx].push_back(std::move(*r));
                } else {
                    auto p = nonlinear_product.multiply(piece);
                    if (!p) {
                        return make_error<BigSplittingField>(p.error());
                    }
                    nonlinear_product = std::move(*p);
                    if (!target_piece) {
                        target_piece = piece;
                        target_i = idx;
                    }
                }
            }
            remaining[idx] = std::move(nonlinear_product);
        }

        if (!target_piece) {
            break;  // every cofactor is fully split into linear factors over `field`
        }

        // Honest degree gate before growing the field.
        auto deg_check = checked_mul_i64(field.degree(), target_piece->degree());
        if (!deg_check || *deg_check > max_degree) {
            return make_error<BigSplittingField>(MathError::not_implemented);
        }
        auto adjoined = adjoin_root(field, *target_piece);
        if (!adjoined) {
            return make_error<BigSplittingField>(adjoined.error());
        }
        const BigNumberField new_field = adjoined->field;
        const BigAlgebraicNumber img = adjoined->old_generator;
        const BigAlgebraicNumber new_root = adjoined->root;

        // Carry every cofactor and every found root forward into the enlarged field.
        for (std::size_t i = 0; i < m; ++i) {
            if (remaining[i].degree() >= 1) {
                auto re = reembed_poly(remaining[i], new_field, img);
                if (!re) {
                    return make_error<BigSplittingField>(re.error());
                }
                remaining[i] = std::move(*re);
            } else {
                remaining[i] = poly_one(new_field);
            }
            for (BigAlgebraicNumber& r : found[i]) {
                auto rr = reembed(r, new_field, img);
                if (!rr) {
                    return make_error<BigSplittingField>(rr.error());
                }
                r = std::move(*rr);
            }
        }

        // Divide the freshly adjoined root out of its cofactor and record it.
        auto reduced = divide_out_root(remaining[target_i], new_root, new_field);
        if (!reduced) {
            return make_error<BigSplittingField>(reduced.error());
        }
        remaining[target_i] = std::move(*reduced);
        found[target_i].push_back(new_root);

        field = new_field;
    }

    // Completeness guard (Rule 32): each input contributes exactly deg_i pairwise-distinct
    // roots in the final field.
    std::vector<std::pair<BigRationalPoly, std::vector<BigAlgebraicNumber>>> roots(m);
    for (std::size_t idx = 0; idx < m; ++idx) {
        if (static_cast<std::int64_t>(found[idx].size()) != irreducibles[idx].degree()) {
            return make_error<BigSplittingField>(MathError::domain_error);
        }
        for (std::size_t a = 0; a < found[idx].size(); ++a) {
            for (std::size_t b = a + 1; b < found[idx].size(); ++b) {
                if (found[idx][a].is_equal(found[idx][b])) {
                    return make_error<BigSplittingField>(MathError::domain_error);  // not distinct
                }
            }
        }
        roots[idx] = std::make_pair(irreducibles[idx], std::move(found[idx]));
    }

    return BigSplittingField{.field = std::move(field), .roots = std::move(roots)};
}

}  // namespace nimblecas
