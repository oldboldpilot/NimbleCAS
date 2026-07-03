// NimbleCAS eigenvalues (ROADMAP 7.10 dependency).
// @author Olumuyiwa Oluwasanmi
//
// Exact spectral primitives over Q: the characteristic polynomial, the rational
// eigenvalues (those eigenvalues that happen to be rational), and a basis for the
// eigenspace of a given rational eigenvalue.
//
// characteristic_polynomial uses the Faddeev-LeVerrier algorithm, which needs only
// Matrix multiply/scale/trace/add over Rational -- no symbolic matrix entries and no
// symbolic determinant of (lambda*I - A). It produces the monic polynomial
// p(lambda) = det(lambda*I - A) of degree n, with coefficients c_0..c_n (c_n = 1).
//
// rational_eigenvalues returns only the RATIONAL roots of that polynomial together with
// their multiplicities. Irrational or complex eigenvalues are deliberately NOT returned:
// representing them exactly needs algebraic-number support, which is out of scope. This
// is an honest exact slice of the spectrum, not an approximation.
//
// eigenvectors_for computes a basis for the null space (kernel) of (A - lambda*I) by
// exact Rational row reduction (RREF), yielding one basis null-vector per free column.

export module nimblecas.eigen;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.roots;

export namespace nimblecas {

// The characteristic polynomial p(lambda) = det(lambda*I - A), monic of degree n, with
// coefficients c_0..c_n (c_n = 1) in ascending order (coefficient(i) is the coefficient
// of lambda^i). Computed by the Faddeev-LeVerrier recurrence over Rational, so the whole
// computation stays inside exact rational arithmetic. Requires a square matrix (otherwise
// domain_error). For the 0x0 matrix the empty product convention gives the constant
// polynomial 1. Any overflow/undefined error from the underlying arithmetic is propagated.
[[nodiscard]] auto characteristic_polynomial(const Matrix& a) -> Result<RationalPoly>;

// The rational eigenvalues of A: the rational roots of the characteristic polynomial,
// each paired with its algebraic multiplicity (root, multiplicity). Eigenvalues that are
// irrational or complex are NOT included -- exact algebraic numbers are out of scope, so
// this returns only the rational portion of the spectrum. Requires a square matrix.
[[nodiscard]] auto rational_eigenvalues(const Matrix& a)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

// A basis for the eigenspace of A belonging to `eigenvalue`: a basis for the null space
// (kernel) of (A - eigenvalue*I), obtained by exact Rational row reduction. There is one
// basis vector per free column; each returned inner vector has length n. If the eigenvalue
// is not actually an eigenvalue (the kernel is trivial) the result is empty. Requires a
// square matrix (otherwise domain_error).
[[nodiscard]] auto eigenvectors_for(const Matrix& a, const Rational& eigenvalue)
    -> Result<std::vector<std::vector<Rational>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Reduce `m` (n rows, n cols, dense Rational) to reduced row echelon form in place using
// exact Rational arithmetic. Returns, for each pivot row 0..rank-1, the column index of
// its pivot (pivot columns are strictly increasing). Propagates any arithmetic error.
[[nodiscard]] auto row_reduce(std::vector<std::vector<Rational>>& m, std::size_t n)
    -> Result<std::vector<std::size_t>> {
    std::vector<std::size_t> pivot_cols;
    std::size_t row = 0;
    for (std::size_t col = 0; col < n && row < n; ++col) {
        // Find a pivot (nonzero entry) in this column at or below `row`.
        std::size_t sel = row;
        while (sel < n && m[sel][col].is_zero()) {
            ++sel;
        }
        if (sel == n) {
            continue;  // free column, no pivot here
        }
        std::swap(m[sel], m[row]);

        // Scale the pivot row so the pivot becomes 1.
        const Rational pivot = m[row][col];
        for (std::size_t j = 0; j < n; ++j) {
            auto scaled = m[row][j].divide(pivot);
            if (!scaled) {
                return make_error<std::vector<std::size_t>>(scaled.error());
            }
            m[row][j] = *scaled;
        }

        // Eliminate this column from every other row.
        for (std::size_t r = 0; r < n; ++r) {
            if (r == row || m[r][col].is_zero()) {
                continue;
            }
            const Rational factor = m[r][col];
            for (std::size_t j = 0; j < n; ++j) {
                auto prod = factor.multiply(m[row][j]);
                if (!prod) {
                    return make_error<std::vector<std::size_t>>(prod.error());
                }
                auto diff = m[r][j].subtract(*prod);
                if (!diff) {
                    return make_error<std::vector<std::size_t>>(diff.error());
                }
                m[r][j] = *diff;
            }
        }

        pivot_cols.push_back(col);
        ++row;
    }
    return pivot_cols;
}

}  // namespace

auto characteristic_polynomial(const Matrix& a) -> Result<RationalPoly> {
    if (!a.is_square()) {
        return make_error<RationalPoly>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        // det(lambda*I - A) over the empty product is 1.
        return RationalPoly::constant(Rational::from_int(1));
    }

    // c[i] is the coefficient of lambda^i; c[n] = 1 (monic leading term).
    std::vector<Rational> c(n + 1, Rational::from_int(0));
    c[n] = Rational::from_int(1);

    const Matrix id = Matrix::identity(n);
    Matrix m = Matrix::zero(n, n);

    // Faddeev-LeVerrier:  for k = 1..n:  M <- A*M + c_{n-k+1} * I ;
    //                                    c_{n-k} = -(1/k) * tr(A*M).
    for (std::size_t k = 1; k <= n; ++k) {
        auto am = a.multiply(m);
        if (!am) {
            return make_error<RationalPoly>(am.error());
        }
        auto shift = id.scale(c[n - k + 1]);
        if (!shift) {
            return make_error<RationalPoly>(shift.error());
        }
        auto next_m = am->add(*shift);
        if (!next_m) {
            return make_error<RationalPoly>(next_m.error());
        }
        m = std::move(*next_m);

        auto am2 = a.multiply(m);
        if (!am2) {
            return make_error<RationalPoly>(am2.error());
        }
        auto tr = am2->trace();
        if (!tr) {
            return make_error<RationalPoly>(tr.error());
        }
        auto over_k = tr->divide(Rational::from_int(static_cast<std::int64_t>(k)));
        if (!over_k) {
            return make_error<RationalPoly>(over_k.error());
        }
        auto coeff = over_k->negate();
        if (!coeff) {
            return make_error<RationalPoly>(coeff.error());
        }
        c[n - k] = *coeff;
    }

    return RationalPoly::from_coeffs(std::move(c));
}

auto rational_eigenvalues(const Matrix& a)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>> {
    auto poly = characteristic_polynomial(a);
    if (!poly) {
        return make_error<std::vector<std::pair<Rational, std::int64_t>>>(poly.error());
    }
    return rational_roots(*poly);
}

auto eigenvectors_for(const Matrix& a, const Rational& eigenvalue)
    -> Result<std::vector<std::vector<Rational>>> {
    if (!a.is_square()) {
        return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return std::vector<std::vector<Rational>>{};
    }

    // Build B = A - eigenvalue*I as a dense Rational matrix.
    std::vector<std::vector<Rational>> b(n, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                auto entry = a.at(i, j).subtract(eigenvalue);
                if (!entry) {
                    return make_error<std::vector<std::vector<Rational>>>(entry.error());
                }
                b[i][j] = *entry;
            } else {
                b[i][j] = a.at(i, j);
            }
        }
    }

    auto pivot_cols = row_reduce(b, n);
    if (!pivot_cols) {
        return make_error<std::vector<std::vector<Rational>>>(pivot_cols.error());
    }

    // Mark which columns are pivots; the rest are free.
    std::vector<bool> is_pivot(n, false);
    for (std::size_t col : *pivot_cols) {
        is_pivot[col] = true;
    }

    // One basis null-vector per free column. For free column f: set x_f = 1 and every
    // other free variable 0; then each pivot row (pivot col p in row r) forces
    // x_p = -b[r][f] because the RREF row reads x_p + b[r][f]*x_f = 0.
    std::vector<std::vector<Rational>> basis;
    for (std::size_t f = 0; f < n; ++f) {
        if (is_pivot[f]) {
            continue;
        }
        std::vector<Rational> v(n, Rational::from_int(0));
        v[f] = Rational::from_int(1);
        for (std::size_t r = 0; r < pivot_cols->size(); ++r) {
            const std::size_t p = (*pivot_cols)[r];
            auto neg = b[r][f].negate();
            if (!neg) {
                return make_error<std::vector<std::vector<Rational>>>(neg.error());
            }
            v[p] = *neg;
        }
        basis.push_back(std::move(v));
    }

    return basis;
}

}  // namespace nimblecas
