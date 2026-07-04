// NimbleCAS exact linear index-1 differential-algebraic equations (DAE).
// @author Olumuyiwa Oluwasanmi
//
// A semi-explicit linear DAE couples a differential block to an algebraic constraint:
//
//     x'(t) = A x(t) + B y(t) + p(t)     (differential part; x has nd components)
//     0     = C x(t) + D y(t) + q(t)     (algebraic part;    y has na components)
//
// where A (nd x nd), B (nd x na), C (na x nd), D (na x na) are EXACT rational matrices and
// the forcings p, q are vectors of truncated power series over Q. Such a system is index 1
// exactly when the constraint Jacobian D is INVERTIBLE: then the algebraic variables are
// determined pointwise by the differential ones,
//
//     y = -D^{-1} (C x + q),
//
// and substituting this into the differential block collapses the DAE to a plain linear ODE
// in x alone,
//
//     x' = (A - B D^{-1} C) x + (p - B D^{-1} q).
//
// We form the reduced coefficient matrix M = A - B D^{-1} C and the reduced forcing vector
// r = p - B D^{-1} q (a matrix-times-series-vector product, exact over Q), solve the linear
// ODE system x' = M x + r with x(0) = x0 EXACTLY via the power-series engine
// nimblecas.ode::solve_first_order_system (the graded Picard/Taylor recursion), and then
// recover y = -D^{-1}(C x + q) coefficient by coefficient.
//
// HONESTY. Everything here is EXACT over Q, returned as truncated power series to the
// requested `order` (terms x^0 .. x^{order-1}); there is no floating point and no rounding.
// The reduction is valid precisely for the LINEAR INDEX-1 case, i.e. D invertible. A
// SINGULAR D means the constraint does not pin down y and the problem has higher index
// (index >= 2), which requires index reduction (repeated differentiation of the constraint)
// that this module does NOT perform — such a system is rejected here with
// MathError::domain_error as the honest boundary of the method. The caller supplies only the
// differential initial data x0; the algebraic initial value is NOT free but DETERMINED by
// consistency, y(0) = -D^{-1}(C x0 + q(0)), and is produced as the x^0 coefficient of the
// returned y. Rule 32 railway: every rational/series/matrix error is propagated; dimension
// mismatches, a non-invertible D, and order 0 surface as MathError::domain_error.

export module nimblecas.dae;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;
import nimblecas.ode;

export namespace nimblecas {

// The exact solution of a linear index-1 DAE as truncated power series over Q: x holds the
// nd differential components and y the na algebraic components, each a PowerSeries of the
// requested order. y is the consistent algebraic trajectory -D^{-1}(C x + q).
struct DaeSolution {
    std::vector<PowerSeries> x;  // differential variables, nd components
    std::vector<PowerSeries> y;  // algebraic variables, na components
};

// Solve the semi-explicit linear index-1 DAE
//     x' = A x + B y + p,     0 = C x + D y + q,     x(0) = x0
// exactly as truncated power series with `order` coefficients (terms x^0 .. x^{order-1}).
//
// Shapes (else MathError::domain_error): A is nd x nd, B is nd x na, C is na x nd, D is
// na x na; p has length nd, q has length na, x0 has length nd; order >= 1. D must be
// INVERTIBLE — a singular D is a higher-index (not index-1) problem and is rejected with
// domain_error (see the module honesty note). The forcing series p, q may carry any order;
// each component is retruncated to `order` (zero-padded or truncated) before use.
//
// Method: reduce to M = A - B D^{-1} C and r = p - B D^{-1} q, solve x' = M x + r with
// x(0) = x0 via nimblecas.ode::solve_first_order_system, then set y = -D^{-1}(C x + q).
// Every error from the rational, matrix, power-series, or ODE layers is propagated.
[[nodiscard]] auto solve_linear_index1_dae(const Matrix& A, const Matrix& B, const Matrix& C,
                                           const Matrix& D, const std::vector<PowerSeries>& p,
                                           const std::vector<PowerSeries>& q,
                                           const std::vector<Rational>& x0, std::size_t order)
    -> Result<DaeSolution>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

using SeriesVec = std::vector<PowerSeries>;

// Retruncate a power series to exactly `order` coefficients: zero-pads a shorter forcing
// (e.g. a low-degree polynomial) or truncates a longer one to the working ring Q[[x]]/(x^order).
[[nodiscard]] auto retruncate(const PowerSeries& s, std::size_t order) -> Result<PowerSeries> {
    auto c = s.coefficients();
    return PowerSeries::from_coeffs(std::vector<Rational>(c.begin(), c.end()), order);
}

// Retruncate every component of a series vector to `order`.
[[nodiscard]] auto retruncate_vec(const SeriesVec& v, std::size_t order) -> Result<SeriesVec> {
    SeriesVec out;
    out.reserve(v.size());
    for (const auto& s : v) {
        auto r = retruncate(s, order);
        if (!r) {
            return make_error<SeriesVec>(r.error());
        }
        out.push_back(std::move(*r));
    }
    return out;
}

// Exact rational-matrix times series-vector: given m (rows x cols) and v (cols components,
// each of order `order`), return the rows-component vector whose i-th entry is
// sum_j m(i,j) * v[j], accumulated in the truncated series ring. Fails with domain_error on
// a shape mismatch and propagates any rational/series error.
[[nodiscard]] auto mat_series_vec(const Matrix& m, const SeriesVec& v, std::size_t order)
    -> Result<SeriesVec> {
    if (m.cols() != v.size()) {
        return make_error<SeriesVec>(MathError::domain_error);
    }
    SeriesVec out;
    out.reserve(m.rows());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        auto seed = PowerSeries::zero(order);
        if (!seed) {
            return make_error<SeriesVec>(seed.error());
        }
        PowerSeries row = std::move(*seed);
        for (std::size_t j = 0; j < m.cols(); ++j) {
            auto term = v[j].scale(m.at(i, j));  // m(i,j) * v[j]
            if (!term) {
                return make_error<SeriesVec>(term.error());
            }
            auto sum = row.add(*term);
            if (!sum) {
                return make_error<SeriesVec>(sum.error());
            }
            row = std::move(*sum);
        }
        out.push_back(std::move(row));
    }
    return out;
}

// Componentwise sum of two equal-length series vectors.
[[nodiscard]] auto vec_add(const SeriesVec& a, const SeriesVec& b) -> Result<SeriesVec> {
    if (a.size() != b.size()) {
        return make_error<SeriesVec>(MathError::domain_error);
    }
    SeriesVec out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto s = a[i].add(b[i]);
        if (!s) {
            return make_error<SeriesVec>(s.error());
        }
        out.push_back(std::move(*s));
    }
    return out;
}

// Componentwise difference of two equal-length series vectors (a - b).
[[nodiscard]] auto vec_subtract(const SeriesVec& a, const SeriesVec& b) -> Result<SeriesVec> {
    if (a.size() != b.size()) {
        return make_error<SeriesVec>(MathError::domain_error);
    }
    SeriesVec out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto d = a[i].subtract(b[i]);
        if (!d) {
            return make_error<SeriesVec>(d.error());
        }
        out.push_back(std::move(*d));
    }
    return out;
}

}  // namespace

auto solve_linear_index1_dae(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D,
                             const SeriesVec& p, const SeriesVec& q, const std::vector<Rational>& x0,
                             std::size_t order) -> Result<DaeSolution> {
    if (order == 0) {
        return make_error<DaeSolution>(MathError::domain_error);
    }
    // Shape validation: A (nd x nd), D (na x na) fix the two block sizes; the off-diagonal
    // blocks and the forcing/initial vectors must be consistent with them.
    if (!A.is_square() || !D.is_square()) {
        return make_error<DaeSolution>(MathError::domain_error);
    }
    const std::size_t nd = A.rows();
    const std::size_t na = D.rows();
    if (B.rows() != nd || B.cols() != na || C.rows() != na || C.cols() != nd) {
        return make_error<DaeSolution>(MathError::domain_error);
    }
    if (p.size() != nd || q.size() != na || x0.size() != nd) {
        return make_error<DaeSolution>(MathError::domain_error);
    }

    // Index-1 pivot: D must be invertible. A singular D (higher index) fails here as
    // domain_error — this module does not perform index reduction.
    auto Dinv = D.inverse();
    if (!Dinv) {
        return make_error<DaeSolution>(Dinv.error());
    }

    // Reduced coefficient matrix M = A - B D^{-1} C.
    auto BDinv = B.multiply(*Dinv);  // nd x na
    if (!BDinv) {
        return make_error<DaeSolution>(BDinv.error());
    }
    auto BDinvC = BDinv->multiply(C);  // nd x nd
    if (!BDinvC) {
        return make_error<DaeSolution>(BDinvC.error());
    }
    auto M = A.subtract(*BDinvC);
    if (!M) {
        return make_error<DaeSolution>(M.error());
    }

    // Bring the forcings into the working ring Q[[x]]/(x^order).
    auto pr = retruncate_vec(p, order);
    if (!pr) {
        return make_error<DaeSolution>(pr.error());
    }
    auto qr = retruncate_vec(q, order);
    if (!qr) {
        return make_error<DaeSolution>(qr.error());
    }

    // Reduced forcing r = p - B D^{-1} q.
    auto BDinvq = mat_series_vec(*BDinv, *qr, order);
    if (!BDinvq) {
        return make_error<DaeSolution>(BDinvq.error());
    }
    auto r = vec_subtract(*pr, *BDinvq);
    if (!r) {
        return make_error<DaeSolution>(r.error());
    }

    // Solve the reduced linear ODE x' = M x + r, x(0) = x0, exactly as truncated series.
    // The vector field is autonomous in the series ring: r is a fixed series vector.
    Matrix Mmat = std::move(*M);
    SeriesVec rvec = std::move(*r);
    SystemOperator field = [Mmat, rvec, order](const SeriesVec& u) -> Result<SeriesVec> {
        auto mx = mat_series_vec(Mmat, u, order);
        if (!mx) {
            return make_error<SeriesVec>(mx.error());
        }
        return vec_add(*mx, rvec);
    };
    auto xsol = solve_first_order_system(std::move(field), x0, order);
    if (!xsol) {
        return make_error<DaeSolution>(xsol.error());
    }

    // Recover the algebraic variables y = -D^{-1}(C x + q) = (-D^{-1}) (C x + q).
    auto Cx = mat_series_vec(C, *xsol, order);  // na components
    if (!Cx) {
        return make_error<DaeSolution>(Cx.error());
    }
    auto Cxq = vec_add(*Cx, *qr);
    if (!Cxq) {
        return make_error<DaeSolution>(Cxq.error());
    }
    auto negDinv = Dinv->scale(Rational::from_int(-1));
    if (!negDinv) {
        return make_error<DaeSolution>(negDinv.error());
    }
    auto ysol = mat_series_vec(*negDinv, *Cxq, order);
    if (!ysol) {
        return make_error<DaeSolution>(ysol.error());
    }

    return DaeSolution{std::move(*xsol), std::move(*ysol)};
}

}  // namespace nimblecas
