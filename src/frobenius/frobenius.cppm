// NimbleCAS Frobenius / rational canonical form (RCF) over the rationals, Q (ROADMAP 7.10).
// @author Olumuyiwa Oluwasanmi
//
// The rational canonical form (a.k.a. Frobenius normal form) of a square matrix A over Q
// is the block-diagonal matrix RCF(A) = diag(C(f_1), ..., C(f_k)), where f_1 | f_2 | ...
// | f_k are the INVARIANT FACTORS of A (monic polynomials over Q[x], each dividing the
// next, with product equal to the characteristic polynomial and f_k equal to the minimal
// polynomial) and C(f) is the companion matrix of f. Unlike the Jordan form, the RCF is
// EXACT over Q: it never needs the eigenvalues, so it is computable with no field
// extension and no floating point -- the entire computation stays inside Q[x].
//
// The engine of this module is the Smith normal form (SNF) of the characteristic matrix
// x*I - A over the principal ideal domain Q[x]. Reducing x*I - A by Q[x] row/column
// operations (the polynomial-degree Euclidean valuation drives the pivoting) to a
// diagonal diag(d_1, ..., d_n) with d_1 | d_2 | ... | d_n (each monic) exposes the
// invariant factors as the non-constant d_i. Because SNF preserves the determinant up to
// a unit and each d_i is normalised monic, the product of the d_i equals the monic
// characteristic polynomial exactly -- the divisibility chain is what makes the result
// canonical rather than merely block-companion.
//
// Honesty (Rule 32): every result is either exact over Q or an honest MathError. A
// non-square matrix is a domain_error; any int64 numerator/denominator overflow from the
// underlying Rational / RationalPoly arithmetic is propagated as overflow. This module
// returns the canonical form together with the invariant factors, NOT the transforming
// change-of-basis matrix P (RCF(A) = P^{-1} A P): computing P exactly is a separate,
// harder task, and returning a plausible-but-wrong P would violate the honesty invariant,
// so it is deliberately omitted rather than faked.

export module nimblecas.frobenius;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// The invariant factors f_1 | f_2 | ... | f_k of A: monic polynomials over Q[x], each
// dividing the next, in ascending order (f_1 the smallest degree, f_k the minimal
// polynomial). Their product is the characteristic polynomial, so the degrees sum to n.
// Computed exactly via the Smith normal form of x*I - A over Q[x]. The 0x0 matrix has no
// invariant factors (an empty list). Requires a square matrix (domain_error otherwise);
// arithmetic overflow is propagated.
[[nodiscard]] auto invariant_factors(const Matrix& a) -> Result<std::vector<RationalPoly>>;

// The minimal polynomial of A: the monic generator of the ideal of polynomials that
// annihilate A, equal to the last (largest) invariant factor. For the 0x0 matrix the
// empty-product convention gives the constant polynomial 1. Requires a square matrix.
[[nodiscard]] auto minimal_polynomial(const Matrix& a) -> Result<RationalPoly>;

// The companion matrix C(p) of a polynomial p of degree n >= 1, in the right-column form:
// ones on the sub-diagonal and the negated coefficients of the monic form of p down the
// last column. Its characteristic polynomial is exactly the monic form of p. The zero
// polynomial and any constant (degree 0) polynomial are a domain_error; a non-monic p is
// accepted and normalised to monic first.
[[nodiscard]] auto companion_matrix(const RationalPoly& p) -> Result<Matrix>;

// The rational canonical (Frobenius) form of A: the block-diagonal matrix
// diag(C(f_1), ..., C(f_k)) of the companion matrices of the invariant factors. Exact
// over Q. The 0x0 matrix maps to itself. Requires a square matrix (domain_error
// otherwise); overflow is propagated. NOTE: this returns the canonical form only -- the
// transforming matrix P with RCF(A) = P^{-1} A P is NOT computed (see the module header).
[[nodiscard]] auto rational_canonical_form(const Matrix& a) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A dense n x n matrix of Q[x] entries, used as the working buffer for the SNF of x*I - A.
using PolyMat = std::vector<std::vector<RationalPoly>>;

// row_i <- row_i - q * row_k, across every column (columns outside the active submatrix
// already hold zeros in the relevant rows, so a full-width sweep is harmless).
[[nodiscard]] auto row_sub_mul(PolyMat& m, std::size_t n, std::size_t i, std::size_t k,
                               const RationalPoly& q) -> Result<void> {
    for (std::size_t j = 0; j < n; ++j) {
        auto prod = q.multiply(m[k][j]);
        if (!prod) {
            return make_error<void>(prod.error());
        }
        auto diff = m[i][j].subtract(*prod);
        if (!diff) {
            return make_error<void>(diff.error());
        }
        m[i][j] = *diff;
    }
    return {};
}

// col_j <- col_j - q * col_k, across every row (rows outside the active submatrix already
// hold zeros in the relevant columns, so a full-height sweep is harmless).
[[nodiscard]] auto col_sub_mul(PolyMat& m, std::size_t n, std::size_t j, std::size_t k,
                               const RationalPoly& q) -> Result<void> {
    for (std::size_t i = 0; i < n; ++i) {
        auto prod = q.multiply(m[i][k]);
        if (!prod) {
            return make_error<void>(prod.error());
        }
        auto diff = m[i][j].subtract(*prod);
        if (!diff) {
            return make_error<void>(diff.error());
        }
        m[i][j] = *diff;
    }
    return {};
}

// row_i <- row_i + row_k: used to fold a submatrix entry that the pivot does not divide
// into the pivot row, so the next Euclidean reduction step produces a smaller pivot.
[[nodiscard]] auto row_add(PolyMat& m, std::size_t n, std::size_t i, std::size_t k)
    -> Result<void> {
    for (std::size_t j = 0; j < n; ++j) {
        auto sum = m[i][j].add(m[k][j]);
        if (!sum) {
            return make_error<void>(sum.error());
        }
        m[i][j] = *sum;
    }
    return {};
}

// Bring the submatrix m[t..][t..] into Smith form at diagonal position t: after this call
// m[t][t] is a monic gcd-style pivot that divides every remaining submatrix entry, and
// its row and column are otherwise zero. Returns true when a pivot was placed, or false
// when the whole remaining submatrix is zero (nothing left to do). Iterates to a fixpoint;
// the polynomial degree strictly decreases on every re-pivot, so it always terminates.
[[nodiscard]] auto reduce_pivot(PolyMat& m, std::size_t n, std::size_t t) -> Result<bool> {
    for (;;) {
        // 1. Locate the minimal-degree nonzero entry in the active submatrix.
        bool found = false;
        std::size_t pi = t;
        std::size_t pj = t;
        std::int64_t best_deg = 0;
        for (std::size_t i = t; i < n; ++i) {
            for (std::size_t j = t; j < n; ++j) {
                if (m[i][j].is_zero()) {
                    continue;
                }
                const std::int64_t d = m[i][j].degree();
                if (!found || d < best_deg) {
                    found = true;
                    best_deg = d;
                    pi = i;
                    pj = j;
                }
            }
        }
        if (!found) {
            return false;  // the remaining submatrix is entirely zero
        }

        // 2. Move the pivot to (t, t) by a row and a column swap.
        if (pi != t) {
            std::swap(m[pi], m[t]);
        }
        if (pj != t) {
            for (std::size_t i = 0; i < n; ++i) {
                std::swap(m[i][pj], m[i][t]);
            }
        }

        bool changed = false;

        // 3. Reduce column t below the pivot: any nonzero remainder is a smaller pivot.
        for (std::size_t i = t + 1; i < n; ++i) {
            if (m[i][t].is_zero()) {
                continue;
            }
            auto dm = m[i][t].divide(m[t][t]);
            if (!dm) {
                return make_error<bool>(dm.error());
            }
            auto r = row_sub_mul(m, n, i, t, dm->quotient);
            if (!r) {
                return make_error<bool>(r.error());
            }
            if (!m[i][t].is_zero()) {
                changed = true;  // remainder left behind -> re-pivot on it
            }
        }
        if (changed) {
            continue;
        }

        // 4. Reduce row t to the right of the pivot, symmetrically.
        for (std::size_t j = t + 1; j < n; ++j) {
            if (m[t][j].is_zero()) {
                continue;
            }
            auto dm = m[t][j].divide(m[t][t]);
            if (!dm) {
                return make_error<bool>(dm.error());
            }
            auto r = col_sub_mul(m, n, j, t, dm->quotient);
            if (!r) {
                return make_error<bool>(r.error());
            }
            if (!m[t][j].is_zero()) {
                changed = true;
            }
        }
        if (changed) {
            continue;
        }

        // 5. Enforce the divisibility chain: the pivot must divide every submatrix entry.
        // If it does not divide m[i][j], fold row i into row t and re-run -- the next
        // reduction yields a strictly smaller pivot, so the invariant d_t | d_{t+1} | ...
        // is forced without ever leaving Q[x].
        bool nondiv = false;
        for (std::size_t i = t + 1; i < n && !nondiv; ++i) {
            for (std::size_t j = t + 1; j < n && !nondiv; ++j) {
                if (m[i][j].is_zero()) {
                    continue;
                }
                auto dm = m[i][j].divide(m[t][t]);
                if (!dm) {
                    return make_error<bool>(dm.error());
                }
                if (!dm->remainder.is_zero()) {
                    auto r = row_add(m, n, t, i);
                    if (!r) {
                        return make_error<bool>(r.error());
                    }
                    nondiv = true;
                }
            }
        }
        if (nondiv) {
            continue;
        }

        // 6. Pivot finalised: normalise it monic (a unit multiple leaves SNF invariant).
        auto mon = m[t][t].monic();
        if (!mon) {
            return make_error<bool>(mon.error());
        }
        m[t][t] = *mon;
        return true;
    }
}

}  // namespace

auto invariant_factors(const Matrix& a) -> Result<std::vector<RationalPoly>> {
    if (!a.is_square()) {
        return make_error<std::vector<RationalPoly>>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return std::vector<RationalPoly>{};  // no invariant factors for the empty operator
    }

    // Build the characteristic matrix M = x*I - A as a dense grid of Q[x] entries:
    // M[i][j] = (i == j ? x : 0) - a(i, j).
    const RationalPoly x = RationalPoly::monomial(Rational::from_int(1), 1);
    PolyMat m(n, std::vector<RationalPoly>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const RationalPoly diag = (i == j) ? x : RationalPoly{};
            auto entry = diag.subtract(RationalPoly::constant(a.at(i, j)));
            if (!entry) {
                return make_error<std::vector<RationalPoly>>(entry.error());
            }
            m[i][j] = *entry;
        }
    }

    // Reduce to Smith normal form, one diagonal pivot at a time.
    for (std::size_t t = 0; t < n; ++t) {
        auto placed = reduce_pivot(m, n, t);
        if (!placed) {
            return make_error<std::vector<RationalPoly>>(placed.error());
        }
        if (!*placed) {
            break;  // remaining submatrix zero (cannot happen for the nonsingular x*I - A)
        }
    }

    // The invariant factors are the non-constant diagonal entries (the unit entries d_i = 1
    // are the trivial factors and are dropped), already monic from reduce_pivot.
    std::vector<RationalPoly> factors;
    for (std::size_t t = 0; t < n; ++t) {
        if (m[t][t].degree() >= 1) {
            factors.push_back(m[t][t]);
        }
    }
    return factors;
}

auto minimal_polynomial(const Matrix& a) -> Result<RationalPoly> {
    auto factors = invariant_factors(a);
    if (!factors) {
        return make_error<RationalPoly>(factors.error());
    }
    if (factors->empty()) {
        return RationalPoly::constant(Rational::from_int(1));  // 0x0: empty product
    }
    return factors->back();  // the largest invariant factor is the minimal polynomial
}

auto companion_matrix(const RationalPoly& p) -> Result<Matrix> {
    if (p.degree() < 1) {
        return make_error<Matrix>(MathError::domain_error);  // zero / constant has no companion
    }
    auto monic = p.monic();
    if (!monic) {
        return make_error<Matrix>(monic.error());
    }
    const std::size_t n = static_cast<std::size_t>(monic->degree());

    // Right-column companion form: ones on the sub-diagonal, and the negated coefficients
    // c_0 .. c_{n-1} of the monic polynomial x^n + c_{n-1} x^{n-1} + ... + c_0 down the last
    // column. Its characteristic polynomial is exactly that monic polynomial.
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 1; i < n; ++i) {
        rows[i][i - 1] = Rational::from_int(1);
    }
    for (std::size_t i = 0; i < n; ++i) {
        auto neg = monic->coefficient(i).negate();
        if (!neg) {
            return make_error<Matrix>(neg.error());
        }
        rows[i][n - 1] = *neg;
    }
    return Matrix::from_rows(std::move(rows));
}

auto rational_canonical_form(const Matrix& a) -> Result<Matrix> {
    auto factors = invariant_factors(a);
    if (!factors) {
        return make_error<Matrix>(factors.error());
    }
    const std::size_t n = a.rows();  // == sum of the invariant-factor degrees

    // Lay the companion block of each invariant factor along the diagonal.
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, Rational::from_int(0)));
    std::size_t off = 0;
    for (const RationalPoly& f : *factors) {
        auto block = companion_matrix(f);
        if (!block) {
            return make_error<Matrix>(block.error());
        }
        const std::size_t bs = block->rows();
        for (std::size_t i = 0; i < bs; ++i) {
            for (std::size_t j = 0; j < bs; ++j) {
                rows[off + i][off + j] = block->at(i, j);
            }
        }
        off += bs;
    }
    return Matrix::from_rows(std::move(rows));
}

}  // namespace nimblecas
