// NimbleCAS Jordan canonical form WITH the transforming matrix P (A = P*J*P^{-1}),
// exact over Q and over a simple quadratic extension field Q(alpha) (ROADMAP §7.10).
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
//   TIER 3  honest refusal — jordan_form returns MathError::not_implemented when an
//           irreducible factor of the char poly has degree >= 3 (a general splitting field,
//           out of scope) OR when two or more DISTINCT irreducible quadratic factors appear
//           (a possibly-composite extension, out of scope). No wrong or decimalized answer
//           is ever produced.
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

// TIER 2 / TIER 3. The Jordan canonical form J and transforming matrix P of A over a
// simple quadratic extension Q(alpha), valid when the characteristic polynomial's only
// non-linear irreducible factor over Q is a single quadratic q(x) (possibly repeated).
// Returns {field, J, P} with A*P == P*J over Q(alpha) and P invertible, verified exactly.
// Fails with:
//   * domain_error — A is not square or is 0x0, OR the char poly splits over Q (no
//     extension is needed — use rational_jordan_form instead);
//   * not_implemented — an irreducible factor of degree >= 3 is present, OR two or more
//     DISTINCT irreducible quadratic factors are present (out of scope: a general or
//     composite splitting field);
//   * overflow — an int64 overflow in the exact arithmetic.
[[nodiscard]] auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan>;

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
                            const S& zero, const S& one) -> std::pair<Mat<S>, Mat<S>> {
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
    return {std::move(j), std::move(p)};
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
    auto [jmat, pmat] = assemble(*groups, n, zero, one);

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

// --- TIER 2 / TIER 3: over Q(alpha) -----------------------------------------

auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan> {
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
    // conjugate pair in a quadratic extension; degree >= 3 -> out of scope (Tier 3).
    std::optional<RationalPoly> quad;   // the single distinct quadratic factor (monic)
    for (const auto& [f, mult] : *factors) {
        const std::int64_t d = f.degree();
        if (d >= 3) {
            return make_error<AlgebraicJordan>(MathError::not_implemented);  // Tier 3
        }
        if (d == 2) {
            auto fm = f.monic();
            if (!fm) {
                return make_error<AlgebraicJordan>(fm.error());
            }
            if (!quad) {
                quad = std::move(*fm);
            } else if (!quad->is_equal(*fm)) {
                // Two distinct quadratic factors: a possibly-composite extension (Tier 3).
                return make_error<AlgebraicJordan>(MathError::not_implemented);
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
    auto [jmat, pmat] = assemble(*groups, n, zero, one);

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

}  // namespace nimblecas
