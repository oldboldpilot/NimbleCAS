// NimbleCAS unbounded invariant factors / minimal polynomial over Q, on BigRational.
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision mirror of nimblecas.frobenius. Where that module carries
// int64 Rational / RationalPoly coefficients and reports MathError::overflow the moment a
// numerator or denominator saturates, this module carries BigRational / BigRationalPoly
// coefficients and has NO such ceiling. It exists because the int64
// nimblecas.frobenius::minimal_polynomial overflows on the rational multiplication matrices
// that arise in splitting-field primitive-element searches: the intermediate Smith-normal-
// form arithmetic (and the final coefficients themselves) can dwarf the ~9.22e18 int64
// envelope. On the bignum tier those values are the exact elements of Q they mathematically
// are, so the search never has to abandon a candidate because the arithmetic ran out of room.
//
// The engine is identical to the int64 tier: the Smith normal form (SNF) of the
// characteristic matrix x*I - A over the principal ideal domain Q[x]. Reducing x*I - A by
// Q[x] row/column operations (the polynomial-degree Euclidean valuation drives the pivoting)
// to a diagonal diag(d_1, ..., d_n) with d_1 | d_2 | ... | d_n (each monic) exposes the
// invariant factors as the non-constant d_i. Their product is the characteristic polynomial,
// the divisibility chain is what makes the result canonical, and the last (largest) factor is
// the minimal polynomial.
//
// Honesty (Rule 32): every result is either exact over Q or an honest MathError. A non-square
// matrix is a domain_error. Unlike the int64 tier there is NO overflow path to propagate:
// BigRational's add/subtract/multiply are infallible (arbitrary precision cannot overflow),
// so the polynomial add/subtract/multiply used below are likewise infallible. The only
// fallible primitives are BigRationalPoly::divide (zero divisor) and monic (zero leading
// coefficient); here the divisor is always the current nonzero pivot, so neither can actually
// fire, but the Result is threaded through for an honest, uniform surface. This module returns
// the invariant factors / minimal polynomial only, NOT any change-of-basis matrix.

export module nimblecas.bigfrobenius;

import std;
import nimblecas.core;
import nimblecas.bigratpoly;
import nimblecas.bigmatrix;

export namespace nimblecas {

// The invariant factors f_1 | f_2 | ... | f_k of A: monic polynomials over Q[x], each
// dividing the next, in ascending order (f_1 the smallest degree, f_k the minimal
// polynomial). Their product is the characteristic polynomial, so the degrees sum to n.
// Computed exactly via the Smith normal form of x*I - A over Q[x] on BigRational, so it
// NEVER overflows. The 0x0 matrix has no invariant factors (an empty list). Requires a
// square matrix (domain_error otherwise).
[[nodiscard]] auto invariant_factors(const BigMatrix& a) -> Result<std::vector<BigRationalPoly>>;

// The minimal polynomial of A: the monic generator of the ideal of polynomials that
// annihilate A, equal to the last (largest) invariant factor. For the 0x0 matrix the
// empty-product convention gives the constant polynomial 1. Requires a square matrix.
// Exact and unbounded: overflow cannot arise on this bignum tier.
[[nodiscard]] auto minimal_polynomial(const BigMatrix& a) -> Result<BigRationalPoly>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A dense n x n matrix of Q[x] entries (BigRational coefficients), used as the working
// buffer for the SNF of x*I - A.
using PolyMat = std::vector<std::vector<BigRationalPoly>>;

// row_i <- row_i - q * row_k, across every column (columns outside the active submatrix
// already hold zeros in the relevant rows, so a full-width sweep is harmless). INFALLIBLE:
// BigRationalPoly multiply/subtract only combine magnitudes and cannot overflow, so unlike
// the int64 frobenius helper there is no error channel to thread.
auto row_sub_mul(PolyMat& m, std::size_t n, std::size_t i, std::size_t k,
                 const BigRationalPoly& q) -> void {
    for (std::size_t j = 0; j < n; ++j) {
        m[i][j] = m[i][j].subtract(q.multiply(m[k][j]));
    }
}

// col_j <- col_j - q * col_k, across every row (rows outside the active submatrix already
// hold zeros in the relevant columns, so a full-height sweep is harmless). INFALLIBLE for
// the same reason as row_sub_mul.
auto col_sub_mul(PolyMat& m, std::size_t n, std::size_t j, std::size_t k,
                 const BigRationalPoly& q) -> void {
    for (std::size_t i = 0; i < n; ++i) {
        m[i][j] = m[i][j].subtract(q.multiply(m[i][k]));
    }
}

// row_i <- row_i + row_k: used to fold a submatrix entry that the pivot does not divide into
// the pivot row, so the next Euclidean reduction step produces a smaller pivot. INFALLIBLE:
// BigRationalPoly add cannot overflow.
auto row_add(PolyMat& m, std::size_t n, std::size_t i, std::size_t k) -> void {
    for (std::size_t j = 0; j < n; ++j) {
        m[i][j] = m[i][j].add(m[k][j]);
    }
}

// Bring the submatrix m[t..][t..] into Smith form at diagonal position t: after this call
// m[t][t] is a monic gcd-style pivot that divides every remaining submatrix entry, and its
// row and column are otherwise zero. Returns true when a pivot was placed, or false when the
// whole remaining submatrix is zero (nothing left to do). Iterates to a fixpoint; the
// polynomial degree strictly decreases on every re-pivot, so it always terminates. Fallible
// only through divide/monic, and only in principle: the divisor is always the current nonzero
// pivot, so neither can actually fail here.
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
            row_sub_mul(m, n, i, t, dm->quotient);
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
            col_sub_mul(m, n, j, t, dm->quotient);
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
                    row_add(m, n, t, i);
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

auto invariant_factors(const BigMatrix& a) -> Result<std::vector<BigRationalPoly>> {
    if (!a.is_square()) {
        return make_error<std::vector<BigRationalPoly>>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return std::vector<BigRationalPoly>{};  // no invariant factors for the empty operator
    }

    // Build the characteristic matrix M = x*I - A as a dense grid of Q[x] entries:
    // M[i][j] = (i == j ? x : 0) - a(i, j). The subtraction is infallible on BigRationalPoly.
    const BigRationalPoly x = BigRationalPoly::monomial(BigRational::from_int(1), 1);
    PolyMat m(n, std::vector<BigRationalPoly>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const BigRationalPoly diag = (i == j) ? x : BigRationalPoly{};
            m[i][j] = diag.subtract(BigRationalPoly::constant(a.at(i, j)));
        }
    }

    // Reduce to Smith normal form, one diagonal pivot at a time.
    for (std::size_t t = 0; t < n; ++t) {
        auto placed = reduce_pivot(m, n, t);
        if (!placed) {
            return make_error<std::vector<BigRationalPoly>>(placed.error());
        }
        if (!*placed) {
            break;  // remaining submatrix zero (cannot happen for the nonsingular x*I - A)
        }
    }

    // The invariant factors are the non-constant diagonal entries (the unit entries d_i = 1
    // are the trivial factors and are dropped), already monic from reduce_pivot.
    std::vector<BigRationalPoly> factors;
    for (std::size_t t = 0; t < n; ++t) {
        if (m[t][t].degree() >= 1) {
            factors.push_back(m[t][t]);
        }
    }
    return factors;
}

auto minimal_polynomial(const BigMatrix& a) -> Result<BigRationalPoly> {
    auto factors = invariant_factors(a);
    if (!factors) {
        return make_error<BigRationalPoly>(factors.error());
    }
    if (factors->empty()) {
        return BigRationalPoly::constant(BigRational::from_int(1));  // 0x0: empty product
    }
    return factors->back();  // the largest invariant factor is the minimal polynomial
}

}  // namespace nimblecas
