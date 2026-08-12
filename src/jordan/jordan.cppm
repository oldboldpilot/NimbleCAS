// NimbleCAS Jordan canonical form WITH the transforming matrix P (A = P*J*P^{-1}),
// exact over Q, over a simple quadratic extension field Q(alpha), and over the general
// splitting field of the characteristic polynomial's non-linear irreducible factors
// (ROADMAP §7.10).
// @author Olumuyiwa Oluwasanmi
//
// The Jordan canonical form J of a square matrix A is block-diagonal, one Jordan block
//
//        [ l 1 0 ... 0 ]
//        [ 0 l 1 ... 0 ]
//   J_k =[ ...         ]      (eigenvalue l on the diagonal, 1 on the superdiagonal)
//        [ 0 0 0 ... 1 ]
//        [ 0 0 0 ... l ]
//
// per Jordan chain, together with an invertible P whose columns are the generalized
// eigenvectors so that A*P == P*J exactly. Unlike the Frobenius / rational canonical form
// (nimblecas.frobenius, which is always exact over Q because it never needs the
// eigenvalues), the Jordan form is built FROM the eigenvalues, so it can only be exact
// where the eigenvalues can be represented exactly. This module delivers three honest
// tiers (Rule 32 throughout — an EXACT verified result or an honest MathError; never a
// wrong P and never a decimalized eigenvalue):
//
//   TIER 1  rational_jordan_form(A) — when the characteristic polynomial SPLITS over Q
//           (every eigenvalue is rational). J and P are computed exactly over Q. The
//           generalized eigenspaces are the null spaces of (A - l*I)^k; Jordan chains are
//           read off by the standard top-down nullity construction; A*P == P*J and the
//           invertibility of P are VERIFIED exactly before returning. This is the primary,
//           must-have tier. A matrix whose char poly does not split over Q is a
//           domain_error here (there is no Jordan form over Q) — use jordan_form instead.
//
//   TIER 2  jordan_form(A) — when the char poly does NOT split over Q but its only
//           non-linear irreducible factor is a SINGLE quadratic q(x) = x^2 + b x + c
//           (possibly repeated), whose two conjugate roots (-b +/- alpha)/... both live in
//           the quadratic extension Q(alpha) = Q[x]/(q). Every eigenvalue — the rational
//           ones and the conjugate pair alpha, (-b - alpha) — is embedded in Q(alpha), and
//           the entire generalized-eigenvector linear algebra (RREF / null space / chains)
//           is carried out exactly over the AlgebraicNumber field. A*P == P*J over Q(alpha)
//           and the invertibility of P are VERIFIED exactly. J and P are returned as
//           matrices of AlgebraicNumber that carry their NumberField (see AlgebraicJordan).
//
//   TIER 3  jordan_form(A) / jordan_form(A, max_field_degree) — when the char poly's
//           non-linear irreducible factors do NOT fit the single-quadratic Tier 2 case (an
//           irreducible factor of degree >= 3 is present, OR two or more DISTINCT
//           irreducible quadratic factors appear), the GENERAL splitting field of every
//           non-linear factor is built via nimblecas.splitfield::splitting_field, capped at
//           `max_field_degree` (default kDefaultMaxSplittingFieldDegree = 12; e.g. a quartic
//           with Galois group S4 needs degree 24 and is refused). Every eigenvalue -- the
//           rational ones and every harvested root of every non-linear factor -- is rebuilt
//           in that ONE common field, and the SAME compute_groups -> assemble -> verify
//           pipeline as Tier 2 produces J and P. Fails honestly with
//           MathError::not_implemented when splitting_field's own envelope is exceeded (a
//           possibility for large or high-Galois-group factors); jordan_structure remains
//           the exact-over-Q fallback in that case -- never a wrong or decimalized answer.
//
// The tier boundary is documented precisely in docs/reference/jordan.md.

module;
#include <cassert>

export module nimblecas.jordan;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.roots;
import nimblecas.eigen;
import nimblecas.factor;
import nimblecas.algnum;
import nimblecas.splitfield;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigalgnum;
import nimblecas.bigsplitfield;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// TIER 1 result — Jordan form and transform over Q.
// ---------------------------------------------------------------------------
// jordan is the block-diagonal Jordan matrix J; transform is the invertible P with
// A*transform == transform*jordan exactly (equivalently transform^{-1}*A*transform == J).
// Both are exact Rational matrices. For the 0x0 matrix both are the empty 0x0 matrix.
struct RationalJordan {
    Matrix jordan;     // J
    Matrix transform;  // P  (A*P == P*J, P invertible)
};

// ---------------------------------------------------------------------------
// TIER 2 result — Jordan form and transform over a quadratic extension Q(alpha).
// ---------------------------------------------------------------------------
// field is the simple extension Q(alpha) = Q[x]/(q) the eigenvalues live in. jordan (J)
// and transform (P) are dense row-major matrices of AlgebraicNumber over that field, with
// A*P == P*J over Q(alpha) exactly and P invertible. Every entry is an exact element of
// the field (a rational-coefficient residue), never a floating-point approximation.
struct AlgebraicJordan {
    NumberField field;
    std::vector<std::vector<AlgebraicNumber>> jordan;     // J, n x n over Q(alpha)
    std::vector<std::vector<AlgebraicNumber>> transform;  // P, n x n over Q(alpha)
};

// TIER 1. The Jordan canonical form J and transforming matrix P of A over Q, valid when
// the characteristic polynomial splits over Q (all eigenvalues rational). Returns {J, P}
// with A*P == P*J and P invertible, both verified exactly. Fails with:
//   * domain_error — A is not square, OR the characteristic polynomial does NOT split
//     over Q (some eigenvalue is irrational/complex — there is no Jordan form over Q; try
//     jordan_form for the single-quadratic-extension case);
//   * overflow — an int64 numerator/denominator overflow in the exact arithmetic.
[[nodiscard]] auto rational_jordan_form(const Matrix& a) -> Result<RationalJordan>;

// The default cap on the degree of the splitting field the TIER 3 general path is willing
// to build (see jordan_form(a, max_field_degree) below). 12 comfortably covers e.g. two or
// three independent quadratics (degree <= 8) or a cubic with an S3 Galois group (degree 6);
// a quartic with Galois group S4 (degree 24) is out of this default's reach.
inline constexpr std::int64_t kDefaultMaxSplittingFieldDegree = 12;

// TIER 2 / TIER 3. The Jordan canonical form J and transforming matrix P of A over the field
// its eigenvalues live in: the simple quadratic extension Q(alpha) when the char poly's only
// non-linear irreducible factor is a single quadratic (possibly repeated) -- TIER 2 -- or
// else the GENERAL splitting field of every non-linear irreducible factor, built via
// nimblecas.splitfield and capped at `max_field_degree` -- TIER 3. Returns {field, J, P}
// with A*P == P*J over `field` and P invertible, verified exactly. The single-argument
// overload uses kDefaultMaxSplittingFieldDegree. Fails with:
//   * domain_error — A is not square or is 0x0, OR the char poly splits over Q (no
//     extension is needed — use rational_jordan_form instead);
//   * not_implemented — building the splitting field of the non-linear factors would exceed
//     `max_field_degree` at some point, or any of splitting_field's own internal budgets is
//     exceeded (an honest refusal — jordan_structure remains the exact-over-Q fallback);
//   * overflow — an int64 overflow in the exact arithmetic.
[[nodiscard]] auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan>;
[[nodiscard]] auto jordan_form(const Matrix& a, std::int64_t max_field_degree)
    -> Result<AlgebraicJordan>;

// ---------------------------------------------------------------------------
// TIER 3 (bignum) result — Jordan form and transform over the UNBOUNDED splitting field.
// ---------------------------------------------------------------------------
// The arbitrary-precision mirror of AlgebraicJordan: field is a BigNumberField (BigRational
// coefficients, no int64 ceiling), and jordan (J) / transform (P) are dense row-major
// matrices of BigAlgebraicNumber with A*P == P*J over `field` exactly and P invertible. This
// tier exists so that splitting fields whose construction OVERFLOWS the int64 AlgebraicJordan
// tier -- the headline case being the degree-6 splitting field of x^3 - 2 -- still yield an
// exact Jordan form. Every entry is an exact element of the field, never a float.
struct BigAlgebraicJordan {
    BigNumberField field;
    std::vector<std::vector<BigAlgebraicNumber>> jordan;     // J, n x n over `field`
    std::vector<std::vector<BigAlgebraicNumber>> transform;  // P, n x n over `field`
};

// TIER 3 over the UNBOUNDED rationals. The bignum mirror of jordan_form's general splitting-
// field path: it builds the splitting field of the characteristic polynomial's non-linear
// irreducible factors on BigRational via nimblecas.bigsplitfield, so it NEVER fails with
// overflow. A matrix whose eigenvalue splitting field the int64 jordan_form cannot construct
// without saturating int64 -- e.g. the companion matrix of x^3 - 2, whose splitting field has
// degree 6 -- is handled exactly here. The characteristic polynomial and its Q-factorization
// are computed on the int64 tier (their coefficients are small, bounded by the entries of A --
// overflow strikes only later, inside the splitting-field arithmetic), and only the non-linear
// factors are lifted to BigRationalPoly for the bignum splitting_field; A and the rational
// eigenvalues are then embedded into that one common field and the SAME
// compute_groups -> assemble -> verify pipeline (S = BigAlgebraicNumber) produces J and P.
// The single-argument overload uses kDefaultMaxSplittingFieldDegree. Fails with:
//   * domain_error — A is not square or is 0x0, OR the char poly splits over Q (no extension
//     is needed — use rational_jordan_form);
//   * not_implemented — the bignum splitting field would exceed `max_field_degree` at some
//     point, or one of bigsplitfield's own internal budgets (Trager's shift search, the
//     primitive-element search, factor_over_Q's budget) is exceeded;
//   * overflow — ONLY from the int64 characteristic_polynomial / factor_over_Q pre-pass, never
//     from the splitting-field construction itself (that is the ceiling this tier removes).
[[nodiscard]] auto jordan_form_bignum(const Matrix& a) -> Result<BigAlgebraicJordan>;
[[nodiscard]] auto jordan_form_bignum(const Matrix& a, std::int64_t max_field_degree)
    -> Result<BigAlgebraicJordan>;

// ---------------------------------------------------------------------------
// jordan_structure — the exact-over-Q Jordan block STRUCTURE (Segre characteristic),
// without constructing any extension field.
// ---------------------------------------------------------------------------
// For each irreducible factor m_i(x) of the characteristic polynomial (degree d_i,
// multiplicity e_i in the characteristic polynomial), EVERY one of the d_i conjugate
// roots of m_i has the SAME Jordan block-size partition of e_i (by Galois symmetry --
// A is rational, so its char poly's irreducible factors are exactly the minimal
// polynomials of Galois orbits, and conjugate roots are indistinguishable to any
// rational invariant such as rank). block_sizes holds that shared partition,
// descending, summing to e_i.
struct JordanFactorStructure {
    RationalPoly factor;   // the irreducible factor m_i(x), as returned by
                            // factor_over_Q (a primitive integer polynomial lifted
                            // into Q[x]; not necessarily monic)
    std::int64_t degree;      // d_i = deg(m_i) -- the number of conjugate roots
    std::int64_t multiplicity;  // e_i -- the multiplicity of m_i in the char poly
    std::vector<std::int64_t> block_sizes;  // descending; sum == multiplicity
};

// The full Jordan block structure of A: one JordanFactorStructure per distinct
// irreducible factor of the characteristic polynomial, in canonical order (factor
// degree ascending, then coefficient-lexicographic -- comparing coefficient(d),
// coefficient(d-1), ..., coefficient(0) in that order -- among equal-degree factors).
// Deterministic regardless of factor_over_Q's unordered return order.
struct JordanStructure {
    std::vector<JordanFactorStructure> factors;
};

// The exact-over-Q Jordan block structure (Segre characteristic) of A, valid for ANY
// square rational matrix -- including one whose eigenvalues are irrational or complex
// -- WITHOUT constructing the splitting field they live in. Complements jordan_form
// (Tiers 1-3 above), which additionally builds a transforming matrix P and therefore
// needs the eigenvalues to be representable (rational, or in a single quadratic
// extension). jordan_structure needs neither: rank is invariant under field
// extension, so the block-size partition shared by an irreducible factor's conjugate
// roots is computable entirely over Q from the ranks of powers of m_i(A).
//
// ALGORITHM. p = characteristic_polynomial(A); factor p over Q via factor_over_Q into
// (m_i, e_i) pairs. For each factor of degree d_i: with M = m_i(A) (Horner
// evaluation over the Matrix ring), nu_k = (n - rank(M^k)) / d_i is the total
// dimension -- shared out evenly over the d_i conjugate roots -- of the generalized
// eigenspace at level k; the standard nullity identity
// #blocks_of_size_k = 2*nu_k - nu_{k-1} - nu_{k+1} (nu_0 = 0) then recovers the Segre
// characteristic common to every conjugate root of m_i.
//
// Fails with:
//   * domain_error -- A is not square; OR (n - rank(M^k)) is not evenly divisible by
//     d_i for some computed power (an honest guard -- Rule 32 -- against ever
//     truncating to a plausible-looking but wrong Segre characteristic; unreachable
//     for a correct characteristic polynomial and exact arithmetic); OR the recovered
//     block sizes do not sum to the factor's multiplicity (the same honesty guard,
//     symmetric case).
//   * overflow -- an int64 numerator/denominator overflow in the exact arithmetic
//     (e.g. matrix powers of m_i(A) growing large on big inputs).
//   * whatever error characteristic_polynomial / factor_over_Q propagate (e.g.
//     not_implemented if factor_over_Q's Kronecker search exceeds its internal
//     divisor-tuple budget).
[[nodiscard]] auto jordan_structure(const Matrix& a) -> Result<JordanStructure>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A dense vector / matrix of field elements S (S is Rational or AlgebraicNumber). Both
// types expose the same value-semantic, Result-returning field interface used below:
// add / subtract / multiply / divide / negate -> Result<S>, and is_zero() -> bool.
template <typename S>
using Vec = std::vector<S>;
template <typename S>
using Mat = std::vector<Vec<S>>;  // row-major, rows()==Mat.size(), each row length n

// M * v, where M is n x n and v has length n. `zero` seeds each accumulator.
template <typename S>
[[nodiscard]] auto mat_vec(const Mat<S>& m, const Vec<S>& v, const S& zero) -> Result<Vec<S>> {
    const std::size_t n = v.size();
    Vec<S> out(n, zero);
    for (std::size_t i = 0; i < n; ++i) {
        S acc = zero;
        for (std::size_t j = 0; j < n; ++j) {
            auto prod = m[i][j].multiply(v[j]);
            if (!prod) {
                return make_error<Vec<S>>(prod.error());
            }
            auto sum = acc.add(*prod);
            if (!sum) {
                return make_error<Vec<S>>(sum.error());
            }
            acc = std::move(*sum);
        }
        out[i] = std::move(acc);
    }
    return out;
}

// A * B, both n x n.
template <typename S>
[[nodiscard]] auto mat_mul(const Mat<S>& a, const Mat<S>& b, const S& zero) -> Result<Mat<S>> {
    const std::size_t n = a.size();
    Mat<S> out(n, Vec<S>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            S acc = zero;
            for (std::size_t k = 0; k < n; ++k) {
                auto prod = a[i][k].multiply(b[k][j]);
                if (!prod) {
                    return make_error<Mat<S>>(prod.error());
                }
                auto sum = acc.add(*prod);
                if (!sum) {
                    return make_error<Mat<S>>(sum.error());
                }
                acc = std::move(*sum);
            }
            out[i][j] = std::move(acc);
        }
    }
    return out;
}

// A basis for the null space (kernel) of the n x n matrix m over the field S, by exact
// Gauss-Jordan (RREF) elimination — one basis vector per free column, each of length n.
// Mirrors nimblecas.eigen::eigenvectors_for but is generic over S. Propagates arithmetic
// errors. An empty result means the kernel is trivial (m is invertible).
template <typename S>
[[nodiscard]] auto null_space(const Mat<S>& min, std::size_t n, const S& zero, const S& one)
    -> Result<std::vector<Vec<S>>> {
    Mat<S> m = min;  // working copy
    std::vector<std::size_t> pivot_cols;
    std::size_t row = 0;
    for (std::size_t col = 0; col < n && row < n; ++col) {
        std::size_t sel = row;
        while (sel < n && m[sel][col].is_zero()) {
            ++sel;
        }
        if (sel == n) {
            continue;  // free column
        }
        std::swap(m[sel], m[row]);
        const S pivot = m[row][col];  // copy before mutating the row
        for (std::size_t j = 0; j < n; ++j) {
            auto q = m[row][j].divide(pivot);
            if (!q) {
                return make_error<std::vector<Vec<S>>>(q.error());
            }
            m[row][j] = std::move(*q);
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (r == row || m[r][col].is_zero()) {
                continue;
            }
            const S factor = m[r][col];  // copy before mutating
            for (std::size_t j = 0; j < n; ++j) {
                auto prod = factor.multiply(m[row][j]);
                if (!prod) {
                    return make_error<std::vector<Vec<S>>>(prod.error());
                }
                auto diff = m[r][j].subtract(*prod);
                if (!diff) {
                    return make_error<std::vector<Vec<S>>>(diff.error());
                }
                m[r][j] = std::move(*diff);
            }
        }
        pivot_cols.push_back(col);
        ++row;
    }
    std::vector<bool> is_pivot(n, false);
    for (std::size_t c : pivot_cols) {
        is_pivot[c] = true;
    }
    std::vector<Vec<S>> basis;
    for (std::size_t f = 0; f < n; ++f) {
        if (is_pivot[f]) {
            continue;
        }
        Vec<S> v(n, zero);
        v[f] = one;
        for (std::size_t r = 0; r < pivot_cols.size(); ++r) {
            auto neg = m[r][f].negate();
            if (!neg) {
                return make_error<std::vector<Vec<S>>>(neg.error());
            }
            v[pivot_cols[r]] = std::move(*neg);
        }
        basis.push_back(std::move(v));
    }
    return basis;
}

// An incremental linear-independence sieve over the field S, maintaining its accepted
// vectors in reduced row-echelon form (each has a unique leading pivot column, normalised
// to `one`, and zero at every other accepted pivot). reduce_add(v) reduces v against the
// current set: if the residual is zero, v is dependent (false); otherwise the normalised
// residual is stored and returned as an accepted, honestly-independent vector (true). This
// drives the Jordan-chain construction: feeding lower subspaces first, then a larger
// subspace, the survivors are exactly the new chain generators.
template <typename S>
struct RowSieve {
    std::size_t n;
    S zero;
    S one;
    std::vector<Vec<S>> rows;          // accepted vectors, reduced echelon form
    std::vector<std::size_t> pivots;   // pivot column of each accepted vector

    [[nodiscard]] auto reduce_add(const Vec<S>& in) -> Result<std::pair<bool, Vec<S>>> {
        using R = std::pair<bool, Vec<S>>;
        Vec<S> v = in;
        // Eliminate every existing pivot column from v.
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const S c = v[pivots[i]];  // copy before mutating v
            if (c.is_zero()) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                auto prod = c.multiply(rows[i][j]);
                if (!prod) {
                    return make_error<R>(prod.error());
                }
                auto diff = v[j].subtract(*prod);
                if (!diff) {
                    return make_error<R>(diff.error());
                }
                v[j] = std::move(*diff);
            }
        }
        // Locate the leading nonzero of the residual.
        std::size_t pc = n;
        for (std::size_t j = 0; j < n; ++j) {
            if (!v[j].is_zero()) {
                pc = j;
                break;
            }
        }
        if (pc == n) {
            return R{false, std::move(v)};  // dependent
        }
        // Normalise so v[pc] == one.
        const S pivot = v[pc];  // copy before mutating v
        for (std::size_t j = 0; j < n; ++j) {
            auto q = v[j].divide(pivot);
            if (!q) {
                return make_error<R>(q.error());
            }
            v[j] = std::move(*q);
        }
        // Back-substitute: clear column pc from previously accepted vectors.
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const S c = rows[i][pc];  // copy before mutating the row
            if (c.is_zero()) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                auto prod = c.multiply(v[j]);
                if (!prod) {
                    return make_error<R>(prod.error());
                }
                auto diff = rows[i][j].subtract(*prod);
                if (!diff) {
                    return make_error<R>(diff.error());
                }
                rows[i][j] = std::move(*diff);
            }
        }
        rows.push_back(v);
        pivots.push_back(pc);
        return R{true, std::move(v)};  // independent; residual is a valid new generator
    }
};

// The Jordan chains of the eigenvalue whose shifted operator is N = (A - l*I), over the
// field S. Returns a list of chains; each chain is an ordered list of column vectors
// [ N^{len-1} g, ..., N g, g ] — i.e. the eigenvector first and the top generator last —
// so laying the columns down in this order gives a Jordan block with l on the diagonal and
// 1 on the superdiagonal (A*col_{k} == l*col_{k} + col_{k-1}). Uses the standard top-down
// nullity construction: for k = p (the index, largest block) down to 1, the vectors of
// ker(N^k) that are independent modulo ker(N^{k-1}) AND modulo the images carried down from
// longer chains are exactly the generators of the chains of length k.
template <typename S>
[[nodiscard]] auto jordan_chains(const Mat<S>& n_mat, std::size_t n, const S& zero,
                                 const S& one) -> Result<std::vector<std::vector<Vec<S>>>> {
    using Chains = std::vector<std::vector<Vec<S>>>;
    if (n == 0) {
        return Chains{};
    }
    // kernels[i] is a basis of ker(N^{i+1}); grow the power until the nullity stabilises.
    std::vector<std::vector<Vec<S>>> kernels;
    auto k1 = null_space(n_mat, n, zero, one);
    if (!k1) {
        return make_error<Chains>(k1.error());
    }
    kernels.push_back(std::move(*k1));
    std::size_t last_dim = kernels.back().size();
    Mat<S> power = n_mat;  // N^p, p starting at 1
    std::size_t p = 1;
    while (p < n) {
        auto nxt = mat_mul(power, n_mat, zero);  // N^{p+1}
        if (!nxt) {
            return make_error<Chains>(nxt.error());
        }
        power = std::move(*nxt);
        auto ker = null_space(power, n, zero, one);
        if (!ker) {
            return make_error<Chains>(ker.error());
        }
        if (ker->size() == last_dim) {
            break;  // ker(N^{p+1}) == ker(N^p): the generalized eigenspace is complete
        }
        last_dim = ker->size();
        kernels.push_back(std::move(*ker));
        ++p;
    }
    // kernels[k-1] == basis of ker(N^k) for k = 1..p; ker(N^0) is {0} (empty basis).
    Chains result;
    std::vector<Vec<S>> carried;  // level-k occupants inherited from longer chains
    for (std::size_t k = p; k >= 1; --k) {
        RowSieve<S> sieve{n, zero, one, {}, {}};
        // Seed with a basis of ker(N^{k-1}) ...
        if (k >= 2) {
            for (const Vec<S>& b : kernels[k - 2]) {
                auto r = sieve.reduce_add(b);
                if (!r) {
                    return make_error<Chains>(r.error());
                }
            }
        }
        // ... then the images carried down from longer chains.
        for (const Vec<S>& c : carried) {
            auto r = sieve.reduce_add(c);
            if (!r) {
                return make_error<Chains>(r.error());
            }
        }
        // The survivors among ker(N^k) are the new length-k chain generators.
        std::vector<Vec<S>> new_gens;
        for (const Vec<S>& b : kernels[k - 1]) {
            auto r = sieve.reduce_add(b);
            if (!r) {
                return make_error<Chains>(r.error());
            }
            if (r->first) {
                new_gens.push_back(std::move(r->second));
            }
        }
        // Carry every current level-k occupant (old carried + new generators) down one
        // level via N, for the next (smaller-k) iteration.
        std::vector<Vec<S>> next_carried;
        for (const Vec<S>& x : carried) {
            auto nx = mat_vec(n_mat, x, zero);
            if (!nx) {
                return make_error<Chains>(nx.error());
            }
            next_carried.push_back(std::move(*nx));
        }
        // Build the full chain of each new generator: g, N g, ..., N^{k-1} g, then reverse
        // to eigenvector-first order.
        for (const Vec<S>& g : new_gens) {
            std::vector<Vec<S>> chain;
            chain.push_back(g);
            Vec<S> cur = g;
            for (std::size_t j = 1; j < k; ++j) {
                auto nx = mat_vec(n_mat, cur, zero);
                if (!nx) {
                    return make_error<Chains>(nx.error());
                }
                cur = std::move(*nx);
                chain.push_back(cur);
            }
            // g sits at level k; carry N g down as this chain's level-(k-1) occupant.
            if (chain.size() >= 2) {
                next_carried.push_back(chain[1]);
            }
            std::reverse(chain.begin(), chain.end());
            result.push_back(std::move(chain));
        }
        carried = std::move(next_carried);
        if (k == 1) {
            break;  // avoid unsigned wrap-around on the loop counter
        }
    }
    return result;
}

// Per-eigenvalue chains paired with the eigenvalue itself, laid down in the given order.
template <typename S>
struct EigenGroup {
    S eigenvalue;
    std::vector<std::vector<Vec<S>>> chains;
};

// Assemble J (block-diagonal Jordan) and P (columns = chain vectors) from the groups.
// Column (off + i) of P is chain[i]; the corresponding Jordan block carries `eigenvalue`
// on the diagonal and `one` on the superdiagonal within the block.
template <typename S>
[[nodiscard]] auto assemble(const std::vector<EigenGroup<S>>& groups, std::size_t n,
                            const S& zero, const S& one) -> Result<std::pair<Mat<S>, Mat<S>>> {
    // Rule 32: the chains must account for EXACTLY n columns. If the eigenvalue list handed to
    // compute_groups ever double-counts (or omits) a generalized eigenspace, the running column
    // offset would over- or under-run the n-by-n J/P buffers -- the overrun being an
    // out-of-bounds write (UB) that corrupts memory before verify() ever runs. Check the total
    // up front and refuse honestly rather than write past the buffers.
    std::size_t total = 0;
    for (const EigenGroup<S>& g : groups) {
        for (const std::vector<Vec<S>>& chain : g.chains) {
            total += chain.size();
        }
    }
    if (total != n) {
        return make_error<std::pair<Mat<S>, Mat<S>>>(MathError::domain_error);
    }
    Mat<S> j(n, Vec<S>(n, zero));
    Mat<S> p(n, Vec<S>(n, zero));
    std::size_t off = 0;
    for (const EigenGroup<S>& g : groups) {
        for (const std::vector<Vec<S>>& chain : g.chains) {
            const std::size_t len = chain.size();
            for (std::size_t i = 0; i < len; ++i) {
                for (std::size_t r = 0; r < n; ++r) {
                    p[r][off + i] = chain[i][r];
                }
                j[off + i][off + i] = g.eigenvalue;
                if (i + 1 < len) {
                    j[off + i][off + i + 1] = one;
                }
            }
            off += len;
        }
    }
    return std::pair<Mat<S>, Mat<S>>{std::move(j), std::move(p)};
}

// Exact self-verification (Rule 32): confirm A*P == P*J entrywise and that P is invertible
// (its kernel is trivial). Returns true only when BOTH hold, so a caller that gets false
// must refuse to return a result rather than emit an unverified P.
template <typename S>
[[nodiscard]] auto verify(const Mat<S>& a, const Mat<S>& j, const Mat<S>& p, std::size_t n,
                          const S& zero, const S& one) -> Result<bool> {
    auto ap = mat_mul(a, p, zero);
    if (!ap) {
        return make_error<bool>(ap.error());
    }
    auto pj = mat_mul(p, j, zero);
    if (!pj) {
        return make_error<bool>(pj.error());
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t col = 0; col < n; ++col) {
            auto diff = (*ap)[i][col].subtract((*pj)[i][col]);
            if (!diff) {
                return make_error<bool>(diff.error());
            }
            if (!diff->is_zero()) {
                return false;  // A*P != P*J
            }
        }
    }
    auto ker = null_space(p, n, zero, one);
    if (!ker) {
        return make_error<bool>(ker.error());
    }
    return ker->empty();  // trivial kernel <=> P invertible
}

// Compute the full set of eigenvalue groups (chains per eigenvalue) for the shifted
// operators A - l*I over the field S, in the order the eigenvalues are supplied.
template <typename S>
[[nodiscard]] auto compute_groups(const Mat<S>& a, const std::vector<S>& eigenvalues,
                                  std::size_t n, const S& zero, const S& one)
    -> Result<std::vector<EigenGroup<S>>> {
    using Groups = std::vector<EigenGroup<S>>;
    Groups groups;
    for (const S& lambda : eigenvalues) {
        // N = A - lambda*I.
        Mat<S> nmat = a;
        for (std::size_t i = 0; i < n; ++i) {
            auto diff = nmat[i][i].subtract(lambda);
            if (!diff) {
                return make_error<Groups>(diff.error());
            }
            nmat[i][i] = std::move(*diff);
        }
        auto chains = jordan_chains(nmat, n, zero, one);
        if (!chains) {
            return make_error<Groups>(chains.error());
        }
        groups.push_back(EigenGroup<S>{lambda, std::move(*chains)});
    }
    return groups;
}

// TIER 3: the general splitting-field path. Builds the splitting field of every non-linear
// irreducible factor of `charpoly` via nimblecas.splitfield::splitting_field, rebuilds the
// full DISTINCT eigenvalue list (every rational root, plus every harvested root of every
// non-linear factor) in that one common field, embeds A into it, and reuses the exact same
// compute_groups -> assemble -> verify pipeline the single-quadratic Tier 2 path uses
// (S = AlgebraicNumber). Each eigenvalue is listed exactly ONCE regardless of its algebraic
// multiplicity in the characteristic polynomial -- exactly as Tier 1/2 already do (see e.g.
// Tier 2's `eigenvalues.push_back(alpha)` a single time even for a repeated quadratic
// factor): compute_groups' jordan_chains(N = A - lambda*I) recovers that eigenvalue's ENTIRE
// generalized eigenspace (however many Jordan blocks, summing to the full multiplicity) from
// the nullity growth of N, N^2, ... in ONE call, so listing the same eigenvalue twice would
// double-count its blocks. `factors` is factor_over_Q(charpoly)'s own result; splitting_field
// reports each non-linear factor's roots back in the SAME order they were submitted in.
// Fails with:
//   * whatever error splitting_field propagates -- most notably MathError::not_implemented
//     when the splitting field's degree would exceed max_field_degree at any point, or any
//     of splitting_field's own internal budgets (Trager's shift search, the primitive-
//     element search, factor_over_Q's own budget) is exceeded. This is the honest Tier 3
//     boundary: jordan_structure remains the exact-over-Q fallback.
//   * domain_error -- the unreachable self-verification guard (Rule 32), symmetric with the
//     other tiers.
[[nodiscard]] auto splitting_field_jordan_form(
    const Matrix& a, const RationalPoly& charpoly,
    const std::vector<std::pair<RationalPoly, std::int64_t>>& factors, std::size_t n,
    std::int64_t max_field_degree) -> Result<AlgebraicJordan> {
    // Collect the non-linear irreducible factors (degree >= 2), monic, in factor_over_Q's
    // own order; splitting_field reports roots back in this same order (per its contract).
    // Only the DISTINCT factors matter here -- multiplicity plays no role in which
    // eigenvalues get listed (see the function comment above).
    std::vector<RationalPoly> nonlinear;
    nonlinear.reserve(factors.size());
    for (const auto& [f, mult] : factors) {
        (void)mult;
        if (f.degree() >= 2) {
            auto fm = f.monic();
            if (!fm) {
                return make_error<AlgebraicJordan>(fm.error());
            }
            nonlinear.push_back(std::move(*fm));
        }
    }

    auto split = splitting_field(nonlinear, max_field_degree);
    if (!split) {
        // Honest propagation (Rule 32): a not_implemented / overflow here is NOT fabricated
        // into a plausible-looking answer. jordan_structure remains the exact-over-Q
        // fallback for the block STRUCTURE alone.
        return make_error<AlgebraicJordan>(split.error());
    }
    const NumberField field = std::move(split->field);
    const AlgebraicNumber zero = field.zero();
    const AlgebraicNumber one = field.one();

    // A embedded into the splitting field.
    Mat<AlgebraicNumber> amat(n, Vec<AlgebraicNumber>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            amat[i][j] = field.from_rational(a.at(i, j));
        }
    }

    // The full DISTINCT eigenvalue list: the rational roots (from the linear factors)
    // rebuilt in `field`, then every non-linear factor's harvested roots (already IN
    // `field`) -- each listed exactly once (see the function comment above).
    std::vector<AlgebraicNumber> eigenvalues;
    auto roots = rational_roots(charpoly);
    if (!roots) {
        return make_error<AlgebraicJordan>(roots.error());
    }
    for (const auto& [r, mult] : *roots) {
        (void)mult;
        eigenvalues.push_back(field.from_rational(r));
    }
    for (const auto& [factor_poly, harvested] : split->roots) {
        (void)factor_poly;
        for (const AlgebraicNumber& root : harvested) {
            eigenvalues.push_back(root);
        }
    }

    auto groups = compute_groups(amat, eigenvalues, n, zero, one);
    if (!groups) {
        return make_error<AlgebraicJordan>(groups.error());
    }
    auto assembled = assemble(*groups, n, zero, one);
    if (!assembled) {
        return make_error<AlgebraicJordan>(assembled.error());
    }
    auto& [jmat, pmat] = *assembled;

    auto ok = verify(amat, jmat, pmat, n, zero, one);
    if (!ok) {
        return make_error<AlgebraicJordan>(ok.error());
    }
    if (!*ok) {
        // Unreachable for correct exact arithmetic; an honest guard (Rule 32) so a P that
        // fails the A*P == P*J / invertibility certificate is never returned.
        return make_error<AlgebraicJordan>(MathError::domain_error);
    }

    return AlgebraicJordan{field, std::move(jmat), std::move(pmat)};
}

}  // namespace

// --- TIER 1: over Q ---------------------------------------------------------

auto rational_jordan_form(const Matrix& a) -> Result<RationalJordan> {
    if (!a.is_square()) {
        return make_error<RationalJordan>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return RationalJordan{Matrix{}, Matrix{}};  // empty operator: 0x0 J and P
    }

    auto charpoly = characteristic_polynomial(a);
    if (!charpoly) {
        return make_error<RationalJordan>(charpoly.error());
    }
    auto roots = rational_roots(*charpoly);
    if (!roots) {
        return make_error<RationalJordan>(roots.error());
    }
    // The char poly splits over Q iff the rational-root multiplicities sum to n.
    std::int64_t total = 0;
    for (const auto& [r, mult] : *roots) {
        total += mult;
    }
    if (total != static_cast<std::int64_t>(n)) {
        return make_error<RationalJordan>(MathError::domain_error);  // does not split over Q
    }

    const Rational zero = Rational::from_int(0);
    const Rational one = Rational::from_int(1);

    // A as a dense Rational grid.
    Mat<Rational> amat(n, Vec<Rational>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            amat[i][j] = a.at(i, j);
        }
    }

    std::vector<Rational> eigenvalues;
    eigenvalues.reserve(roots->size());
    for (const auto& [r, mult] : *roots) {
        eigenvalues.push_back(r);
    }

    auto groups = compute_groups(amat, eigenvalues, n, zero, one);
    if (!groups) {
        return make_error<RationalJordan>(groups.error());
    }
    auto assembled = assemble(*groups, n, zero, one);
    if (!assembled) {
        return make_error<RationalJordan>(assembled.error());
    }
    auto& [jmat, pmat] = *assembled;

    auto ok = verify(amat, jmat, pmat, n, zero, one);
    if (!ok) {
        return make_error<RationalJordan>(ok.error());
    }
    if (!*ok) {
        // Unreachable for correct exact arithmetic; an honest guard so a P that fails the
        // A*P == P*J / invertibility certificate is never returned (Rule 32).
        return make_error<RationalJordan>(MathError::domain_error);
    }

    auto jout = Matrix::from_rows(jmat);
    if (!jout) {
        return make_error<RationalJordan>(jout.error());
    }
    auto pout = Matrix::from_rows(pmat);
    if (!pout) {
        return make_error<RationalJordan>(pout.error());
    }
    return RationalJordan{std::move(*jout), std::move(*pout)};
}

// --- TIER 2 / TIER 3: over Q(alpha) or the general splitting field ----------

auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan> {
    return jordan_form(a, kDefaultMaxSplittingFieldDegree);
}

auto jordan_form(const Matrix& a, std::int64_t max_field_degree) -> Result<AlgebraicJordan> {
    if (!a.is_square()) {
        return make_error<AlgebraicJordan>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        // No eigenvalues, hence no field to build; the extension form is undefined here.
        return make_error<AlgebraicJordan>(MathError::domain_error);
    }

    auto charpoly = characteristic_polynomial(a);
    if (!charpoly) {
        return make_error<AlgebraicJordan>(charpoly.error());
    }
    auto factors = factor_over_Q(*charpoly);
    if (!factors) {
        return make_error<AlgebraicJordan>(factors.error());
    }

    // Classify the irreducible factors. Degree 1 -> rational eigenvalue; degree 2 -> a
    // conjugate pair in a quadratic extension; degree >= 3, or a second DISTINCT quadratic,
    // hands off to the general splitting-field path (TIER 3) covering every non-linear
    // factor at once.
    std::optional<RationalPoly> quad;   // the single distinct quadratic factor (monic)
    for (const auto& [f, mult] : *factors) {
        const std::int64_t d = f.degree();
        if (d >= 3) {
            return splitting_field_jordan_form(a, *charpoly, *factors, n, max_field_degree);
        }
        if (d == 2) {
            auto fm = f.monic();
            if (!fm) {
                return make_error<AlgebraicJordan>(fm.error());
            }
            if (!quad) {
                quad = std::move(*fm);
            } else if (!quad->is_equal(*fm)) {
                // Two distinct quadratic factors: the general splitting-field path.
                return splitting_field_jordan_form(a, *charpoly, *factors, n, max_field_degree);
            }
        }
    }
    if (!quad) {
        // No quadratic factor: the char poly splits over Q; no extension is needed.
        return make_error<AlgebraicJordan>(MathError::domain_error);
    }

    // Build the quadratic extension Q(alpha) = Q[x]/(quad), quad = x^2 + B x + C monic.
    auto field_res = NumberField::create(*quad);
    if (!field_res) {
        return make_error<AlgebraicJordan>(field_res.error());
    }
    const NumberField field = std::move(*field_res);
    const AlgebraicNumber zero = field.zero();
    const AlgebraicNumber one = field.one();

    auto alpha_res = field.generator();  // a root of quad
    if (!alpha_res) {
        return make_error<AlgebraicJordan>(alpha_res.error());
    }
    const AlgebraicNumber alpha = std::move(*alpha_res);
    // The conjugate root is (-B) - alpha, since the two roots of x^2 + B x + C sum to -B.
    const Rational bcoeff = quad->coefficient(1);
    auto neg_b = bcoeff.negate();
    if (!neg_b) {
        return make_error<AlgebraicJordan>(neg_b.error());
    }
    auto conj_res = field.from_rational(*neg_b).subtract(alpha);
    if (!conj_res) {
        return make_error<AlgebraicJordan>(conj_res.error());
    }
    const AlgebraicNumber conjugate = std::move(*conj_res);

    // A embedded into Q(alpha).
    Mat<AlgebraicNumber> amat(n, Vec<AlgebraicNumber>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            amat[i][j] = field.from_rational(a.at(i, j));
        }
    }

    // Eigenvalues, in a deterministic order: the rational ones (from the rational roots),
    // then alpha, then its conjugate.
    auto roots = rational_roots(*charpoly);
    if (!roots) {
        return make_error<AlgebraicJordan>(roots.error());
    }
    std::vector<AlgebraicNumber> eigenvalues;
    for (const auto& [r, mult] : *roots) {
        eigenvalues.push_back(field.from_rational(r));
    }
    eigenvalues.push_back(alpha);
    eigenvalues.push_back(conjugate);

    auto groups = compute_groups(amat, eigenvalues, n, zero, one);
    if (!groups) {
        return make_error<AlgebraicJordan>(groups.error());
    }
    auto assembled = assemble(*groups, n, zero, one);
    if (!assembled) {
        return make_error<AlgebraicJordan>(assembled.error());
    }
    auto& [jmat, pmat] = *assembled;

    auto ok = verify(amat, jmat, pmat, n, zero, one);
    if (!ok) {
        return make_error<AlgebraicJordan>(ok.error());
    }
    if (!*ok) {
        // Unreachable for correct exact arithmetic; honest guard (Rule 32).
        return make_error<AlgebraicJordan>(MathError::domain_error);
    }

    return AlgebraicJordan{field, std::move(jmat), std::move(pmat)};
}

// --- TIER 3 (bignum): over the UNBOUNDED splitting field --------------------

auto jordan_form_bignum(const Matrix& a) -> Result<BigAlgebraicJordan> {
    return jordan_form_bignum(a, kDefaultMaxSplittingFieldDegree);
}

auto jordan_form_bignum(const Matrix& a, std::int64_t max_field_degree)
    -> Result<BigAlgebraicJordan> {
    if (!a.is_square()) {
        return make_error<BigAlgebraicJordan>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return make_error<BigAlgebraicJordan>(MathError::domain_error);
    }

    // Rational -> BigRational lift (a valid Rational has a positive denominator, so make
    // never actually fires division_by_zero; the Result is threaded for a uniform surface).
    auto to_big = [](const Rational& q) -> Result<BigRational> {
        return BigRational::make(BigInt::from_i64(q.numerator()), BigInt::from_i64(q.denominator()));
    };

    // Char poly and its Q-factorization on the INT64 tier: their coefficients are small
    // (bounded by A's entries), so overflow does not strike here -- it strikes only later,
    // inside the splitting-field arithmetic, which is exactly what this bignum tier removes.
    auto charpoly = characteristic_polynomial(a);
    if (!charpoly) {
        return make_error<BigAlgebraicJordan>(charpoly.error());
    }
    auto factors = factor_over_Q(*charpoly);
    if (!factors) {
        return make_error<BigAlgebraicJordan>(factors.error());
    }

    // The non-linear irreducible factors (degree >= 2), monic, lifted to BigRationalPoly.
    // No non-linear factor => the char poly splits over Q and no extension is needed.
    std::vector<BigRationalPoly> nonlinear;
    for (const auto& [f, mult] : *factors) {
        (void)mult;
        if (f.degree() >= 2) {
            auto fm = f.monic();
            if (!fm) {
                return make_error<BigAlgebraicJordan>(fm.error());
            }
            nonlinear.push_back(BigRationalPoly::from_ratpoly(*fm));
        }
    }
    if (nonlinear.empty()) {
        return make_error<BigAlgebraicJordan>(MathError::domain_error);  // splits over Q
    }

    // Build the splitting field on the UNBOUNDED rationals -- the step the int64
    // AlgebraicJordan tier cannot complete without overflow for cases like x^3 - 2.
    auto split = splitting_field(std::span<const BigRationalPoly>(nonlinear), max_field_degree);
    if (!split) {
        // Honest propagation (Rule 32): a not_implemented here is NOT fabricated into a
        // plausible answer. jordan_structure remains the exact-over-Q block-structure fallback.
        return make_error<BigAlgebraicJordan>(split.error());
    }
    const BigNumberField field = std::move(split->field);
    const BigAlgebraicNumber zero = field.zero();
    const BigAlgebraicNumber one = field.one();

    // A embedded into the splitting field (each rational entry lifted to BigRational).
    Mat<BigAlgebraicNumber> amat(n, Vec<BigAlgebraicNumber>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto be = to_big(a.at(i, j));
            if (!be) {
                return make_error<BigAlgebraicJordan>(be.error());
            }
            amat[i][j] = field.from_bigrational(*be);
        }
    }

    // The full DISTINCT eigenvalue list: the rational roots rebuilt in `field`, then every
    // non-linear factor's harvested roots (already IN `field`) -- each listed exactly once,
    // exactly as splitting_field_jordan_form does (compute_groups recovers each eigenvalue's
    // ENTIRE generalized eigenspace from the nullity growth of N = A - lambda*I in one call).
    std::vector<BigAlgebraicNumber> eigenvalues;
    auto roots = rational_roots(*charpoly);
    if (!roots) {
        return make_error<BigAlgebraicJordan>(roots.error());
    }
    for (const auto& [r, mult] : *roots) {
        (void)mult;
        auto br = to_big(r);
        if (!br) {
            return make_error<BigAlgebraicJordan>(br.error());
        }
        eigenvalues.push_back(field.from_bigrational(*br));
    }
    for (const auto& [factor_poly, harvested] : split->roots) {
        (void)factor_poly;
        for (const BigAlgebraicNumber& root : harvested) {
            eigenvalues.push_back(root);
        }
    }

    auto groups = compute_groups(amat, eigenvalues, n, zero, one);
    if (!groups) {
        return make_error<BigAlgebraicJordan>(groups.error());
    }
    auto assembled = assemble(*groups, n, zero, one);
    if (!assembled) {
        return make_error<BigAlgebraicJordan>(assembled.error());
    }
    auto& [jmat, pmat] = *assembled;

    auto ok = verify(amat, jmat, pmat, n, zero, one);
    if (!ok) {
        return make_error<BigAlgebraicJordan>(ok.error());
    }
    if (!*ok) {
        // Unreachable for correct exact arithmetic; an honest guard (Rule 32) so a P that
        // fails the A*P == P*J / invertibility certificate is never returned.
        return make_error<BigAlgebraicJordan>(MathError::domain_error);
    }

    return BigAlgebraicJordan{field, std::move(jmat), std::move(pmat)};
}

// --- jordan_structure: exact-over-Q Segre characteristic, no extension field --------

namespace {

// a < b for exact Rationals, via a 128-bit cross-multiply (both denominators are
// canonically positive, so no sign flip is needed and the products cannot overflow
// int64*int64 widened to __int128). Used only to order jordan_structure's output
// canonically -- never to compute a returned value -- so this stays outside the
// checked-Rational arithmetic surface deliberately.
[[nodiscard]] auto rational_less(const Rational& a, const Rational& b) -> bool {
    const __int128 lhs = static_cast<__int128>(a.numerator()) * static_cast<__int128>(b.denominator());
    const __int128 rhs = static_cast<__int128>(b.numerator()) * static_cast<__int128>(a.denominator());
    return lhs < rhs;
}

// Canonical factor order: degree ascending, then coefficient-lexicographic --
// coefficient(d), coefficient(d-1), ..., coefficient(0), highest-degree term first --
// among factors of equal degree.
[[nodiscard]] auto factor_less(const RationalPoly& x, const RationalPoly& y) -> bool {
    const std::int64_t dx = x.degree();
    const std::int64_t dy = y.degree();
    if (dx != dy) {
        return dx < dy;
    }
    for (std::int64_t i = dx; i >= 0; --i) {
        const Rational cx = x.coefficient(static_cast<std::size_t>(i));
        const Rational cy = y.coefficient(static_cast<std::size_t>(i));
        if (!(cx == cy)) {
            return rational_less(cx, cy);
        }
    }
    return false;  // equal
}

// Horner evaluation of poly(A), n x n, using only the exported Matrix API
// (identity / scale / multiply / add). poly is never the zero polynomial here (it is
// always an irreducible factor of a characteristic polynomial, degree >= 1).
[[nodiscard]] auto matrix_poly_eval(const RationalPoly& poly, const Matrix& a, std::size_t n)
    -> Result<Matrix> {
    const std::int64_t deg = poly.degree();
    if (deg < 0) {
        return Matrix::zero(n, n);
    }
    const Matrix id = Matrix::identity(n);
    auto lead = id.scale(poly.coefficient(static_cast<std::size_t>(deg)));
    if (!lead) {
        return make_error<Matrix>(lead.error());
    }
    Matrix m = std::move(*lead);
    for (std::int64_t i = deg - 1; i >= 0; --i) {
        auto prod = m.multiply(a);
        if (!prod) {
            return make_error<Matrix>(prod.error());
        }
        auto shift = id.scale(poly.coefficient(static_cast<std::size_t>(i)));
        if (!shift) {
            return make_error<Matrix>(shift.error());
        }
        auto sum = prod->add(*shift);
        if (!sum) {
            return make_error<Matrix>(sum.error());
        }
        m = std::move(*sum);
    }
    return m;
}

}  // namespace

auto jordan_structure(const Matrix& a) -> Result<JordanStructure> {
    if (!a.is_square()) {
        return make_error<JordanStructure>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return JordanStructure{};  // no eigenvalues: the empty structure
    }

    auto charpoly = characteristic_polynomial(a);
    if (!charpoly) {
        return make_error<JordanStructure>(charpoly.error());
    }
    auto factors = factor_over_Q(*charpoly);
    if (!factors) {
        return make_error<JordanStructure>(factors.error());
    }

    std::vector<std::pair<RationalPoly, std::int64_t>> sorted = std::move(*factors);
    std::ranges::sort(sorted, [](const auto& lhs, const auto& rhs) {
        return factor_less(lhs.first, rhs.first);
    });

    JordanStructure out;
    out.factors.reserve(sorted.size());
    for (const auto& [m, e] : sorted) {
        const std::int64_t d = m.degree();
        if (d <= 0) {
            // An irreducible factor of a characteristic polynomial is never a unit or
            // the zero polynomial; an honest guard against a malformed factor list.
            return make_error<JordanStructure>(MathError::domain_error);
        }

        auto m1 = matrix_poly_eval(m, a, n);
        if (!m1) {
            return make_error<JordanStructure>(m1.error());
        }

        // nu[0] = 0; nu[k] = (n - rank(M^k)) / d for k = 1, 2, ..., grown until it
        // stops increasing (ker(M^k) == ker(M^{k-1})). p (the largest Jordan-block
        // size shared by every conjugate root of m) is bounded by e -- since
        // d*e <= n -- so this converges within n+1 iterations for any correct input;
        // a safety cap of n+1 turns a hypothetical non-convergence into an honest
        // error rather than silently trusting an unstabilised tail.
        std::vector<std::int64_t> nu{0};
        Matrix power = *m1;  // M^1
        bool converged = false;
        for (std::size_t k = 1; k <= n + 1 && !converged; ++k) {
            const std::int64_t r = power.rank();
            const std::int64_t num = static_cast<std::int64_t>(n) - r;
            if (num % d != 0) {
                // HONESTY GUARD (Rule 32): never truncate-divide to a plausible-
                // looking but wrong Segre characteristic.
                return make_error<JordanStructure>(MathError::domain_error);
            }
            const std::int64_t nu_k = num / d;
            nu.push_back(nu_k);
            if (nu_k == nu[nu.size() - 2]) {
                converged = true;
                break;
            }
            auto next = power.multiply(*m1);
            if (!next) {
                return make_error<JordanStructure>(next.error());
            }
            power = std::move(*next);
        }
        if (!converged) {
            return make_error<JordanStructure>(MathError::domain_error);
        }

        // nu now holds nu_0..nu_{p+1} with nu_{p+1} == nu_p (p = largest block size).
        const std::size_t p = nu.size() - 2;
        std::vector<std::int64_t> block_sizes;
        std::int64_t total = 0;
        for (std::size_t k = 1; k <= p; ++k) {
            const std::int64_t count = 2 * nu[k] - nu[k - 1] - nu[k + 1];
            if (count < 0) {
                // HONESTY GUARD (Rule 32): a negative block count is impossible for a
                // correct nullity sequence; refuse rather than emit a nonsensical
                // partition.
                return make_error<JordanStructure>(MathError::domain_error);
            }
            for (std::int64_t c = 0; c < count; ++c) {
                block_sizes.push_back(static_cast<std::int64_t>(k));
            }
            total += count * static_cast<std::int64_t>(k);
        }
        if (total != e) {
            // HONESTY GUARD (Rule 32): the recovered partition must sum to the
            // factor's multiplicity in the characteristic polynomial.
            return make_error<JordanStructure>(MathError::domain_error);
        }
        std::ranges::sort(block_sizes, std::greater<>{});

        out.factors.push_back(
            JordanFactorStructure{m, d, e, std::move(block_sizes)});
    }

    return out;
}

}  // namespace nimblecas
