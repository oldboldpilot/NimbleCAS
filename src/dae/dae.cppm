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

// ===========================================================================
// Higher-index LINEAR CONSTANT-COEFFICIENT DAEs via index reduction.
// ===========================================================================
//
// General linear constant-coefficient DAE (a matrix pencil driven by a forcing series):
//
//     E x'(t) = A x(t) + f(t),     x(0) = x0,
//
// with E, A EXACT rational n x n matrices (E MAY BE SINGULAR) and f a vector of truncated
// power series over Q. When E is invertible this is already an ODE (index 0); when it is
// singular the constraints hidden in the zero rows of E must be DIFFERENTIATED to expose the
// dynamics. We reduce the index by the classical SHUFFLE ALGORITHM (Gantmacher): at each step
// split off the left null space W of E (the rows w with w E = 0), giving the algebraic
// constraints (W A) x + (W f) = 0; differentiate them, (W A) x' = -(W f)', and replace the
// zero rows of the pencil with W A while carrying the surviving differential rows of E along.
// Each pass raises the rank of E; the number of passes needed until E becomes invertible is
// the DIFFERENTIATION INDEX of the DAE (index 0 = already an ODE). Once E is invertible the
// system is a plain linear ODE x' = E^{-1} A x + E^{-1} f, solved EXACTLY over Q by the
// power-series engine nimblecas.ode::solve_first_order_system, exactly as the index-1 path
// reduces onto ode.
//
// HIDDEN CONSTRAINTS / CONSISTENCY. Every constraint (W A) x + (W f) = 0 accumulated during
// reduction must hold along the true trajectory, hence at t = 0 it constrains x0. The union of
// these rows defines the affine CONSISTENCY MANIFOLD of admissible initial vectors. We expose
// whether a given x0 lies on it, and can orthogonally PROJECT x0 onto it (exactly over Q).
//
// HONESTY. This is scoped to LINEAR CONSTANT-COEFFICIENT DAEs with a REGULAR pencil (that is,
// det(sE - A) is not identically zero). NONLINEAR or VARIABLE-COEFFICIENT higher-index DAEs
// are OUT OF SCOPE and are not attempted here (a caller reaching for them should treat this as
// not_implemented). A genuinely SINGULAR / non-regular pencil (the shuffle never terminates:
// E stays singular after n passes) is rejected with MathError::domain_error rather than
// returning a wrong answer. The DIFFERENTIATION INDEX used is the number of constraint
// differentiations to reach an ODE. Index reduction differentiates f, so exact truncated
// results to `order` require f to be sufficiently smooth and supplied to sufficient order:
// polynomial / terminating forcing (as nimblecas.ode supports) captured within `order` is
// exact; a forcing supplied at too low an order loses its highest coefficients under
// differentiation (documented, not silently wrong). Everything is EXACT over Q — no floats.

// How to treat an initial vector that does not lie on the DAE's hidden constraint manifold.
enum class ConsistencyPolicy : std::uint8_t {
    require,     // reject an inconsistent initial vector with MathError::domain_error
    project,     // orthogonally project it (over Q) onto the manifold, then solve
    underlying,  // solve the underlying ODE from the given vector as-is (may leave the manifold)
};

// Exact solution of a general linear constant-coefficient DAE obtained by index reduction.
// x holds the n state components as truncated power series to the requested order; index is
// the computed differentiation index (0 = the pencil is already an ODE, 1 = index-1, ...);
// consistent reports whether the ORIGINALLY supplied initial vector satisfied every hidden
// constraint at t = 0 (before any projection under ConsistencyPolicy::project).
struct LinearDaeSolution {
    std::vector<PowerSeries> x;  // n state components
    std::size_t index{0};        // differentiation index
    bool consistent{true};       // did the supplied x0 lie on the constraint manifold?
};

// Split solution of a possibly higher-index SEMI-EXPLICIT linear DAE (same block layout as
// solve_linear_index1_dae, but D need NOT be invertible): x are the nd differential and y the
// na algebraic components, plus the computed index and the consistency of the supplied guess.
struct SemiExplicitDaeSolution {
    std::vector<PowerSeries> x;  // differential components, nd
    std::vector<PowerSeries> y;  // algebraic components, na
    std::size_t index{0};
    bool consistent{true};
};

// The differentiation index of the regular matrix pencil (E, A): the number of times the
// algebraic constraints must be differentiated to expose an explicit ODE for x'. E and A must
// be square and equal-sized (else domain_error). Index 0 means E is already invertible. A
// non-regular pencil (det(sE - A) identically zero — the shuffle fails to terminate) yields
// domain_error. Forcing-independent.
[[nodiscard]] auto linear_dae_index(const Matrix& E, const Matrix& A) -> Result<std::size_t>;

// Whether x0 is a consistent initial value for E x' = A x + f, i.e. it satisfies every hidden
// algebraic constraint accumulated during index reduction, evaluated at t = 0. Shapes: E, A
// are n x n, f and x0 have length n, order >= 1 (else domain_error); non-regular pencil is
// domain_error.
[[nodiscard]] auto linear_dae_is_consistent(const Matrix& E, const Matrix& A,
                                            const std::vector<PowerSeries>& f,
                                            const std::vector<Rational>& x0, std::size_t order)
    -> Result<bool>;

// Orthogonally project x0 (exactly over Q) onto the affine constraint manifold of
// E x' = A x + f at t = 0, returning the nearest consistent initial vector; an already
// consistent x0 is returned unchanged. Shapes / non-regular pencil errors as above.
[[nodiscard]] auto project_to_consistent(const Matrix& E, const Matrix& A,
                                          const std::vector<PowerSeries>& f,
                                          const std::vector<Rational>& x0, std::size_t order)
    -> Result<std::vector<Rational>>;

// Solve the general linear constant-coefficient DAE E x' = A x + f(t), x(0) = x0, exactly as
// truncated power series with `order` coefficients (terms x^0 .. x^{order-1}), by index
// reduction (the shuffle algorithm) onto nimblecas.ode. Shapes (else domain_error): E, A are
// n x n; f and x0 have length n; order >= 1. E may be singular. `policy` selects how an
// inconsistent x0 is handled (require / project / underlying). Returns the state trajectory,
// the computed differentiation index, and whether the supplied x0 was consistent. A
// non-regular pencil is rejected with domain_error; every rational/series/matrix/ode error is
// propagated.
[[nodiscard]] auto solve_linear_dae(const Matrix& E, const Matrix& A,
                                    const std::vector<PowerSeries>& f,
                                    const std::vector<Rational>& x0, std::size_t order,
                                    ConsistencyPolicy policy = ConsistencyPolicy::require)
    -> Result<LinearDaeSolution>;

// Solve a possibly HIGHER-INDEX semi-explicit linear DAE
//     x' = A x + B y + p,     0 = C x + D y + q,     [x; y](0) = [x0; y0]
// (the block layout of solve_linear_index1_dae, but with NO requirement that D be invertible).
// It is embedded as the general DAE E z' = M z + f with z = [x; y], E = diag(I_nd, 0_na),
// M = [[A, B], [C, D]], f = [p; q], solved by solve_linear_dae, then split back into x and y.
// Because higher index constrains the initial data, the FULL guess [x0; y0] is supplied and,
// under the default ConsistencyPolicy::project, projected onto the constraint manifold (so a
// merely approximate y0 guess is corrected exactly). Shapes as in solve_linear_index1_dae with
// y0 of length na; order >= 1. Non-regular pencil / shape errors are domain_error.
[[nodiscard]] auto solve_semiexplicit_dae(const Matrix& A, const Matrix& B, const Matrix& C,
                                          const Matrix& D, const std::vector<PowerSeries>& p,
                                          const std::vector<PowerSeries>& q,
                                          const std::vector<Rational>& x0,
                                          const std::vector<Rational>& y0, std::size_t order,
                                          ConsistencyPolicy policy = ConsistencyPolicy::project)
    -> Result<SemiExplicitDaeSolution>;

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

// ---------------------------------------------------------------------------
// Exact-over-Q linear-algebra helpers for index reduction (the shuffle algorithm).
// ---------------------------------------------------------------------------

using RatRows = std::vector<std::vector<Rational>>;

// Copy a Matrix into row-of-rationals form for elimination scratch work.
[[nodiscard]] auto mat_to_rows(const Matrix& m) -> RatRows {
    RatRows rows;
    rows.reserve(m.rows());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        std::vector<Rational> row;
        row.reserve(m.cols());
        for (std::size_t j = 0; j < m.cols(); ++j) {
            row.push_back(m.at(i, j));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// Rank of a set of rational rows (all of common width): built through Matrix::rank over Q.
[[nodiscard]] auto ratrows_rank(RatRows rows) -> std::int64_t {
    if (rows.empty()) {
        return 0;
    }
    auto m = Matrix::from_rows(std::move(rows));
    if (!m) {
        return 0;  // ragged never happens on our uniform rows; be conservative
    }
    return m->rank();
}

// Reduced row echelon form of `a` (a.size() rows, `cols` columns) in place over Q, with
// partial pivoting on nonzero pivots. Returns the list of pivot columns in row order (so pivot
// i sits in row i). Overflow in the exact rational arithmetic is propagated.
[[nodiscard]] auto rref(RatRows& a, std::size_t cols) -> Result<std::vector<std::size_t>> {
    const std::size_t rows = a.size();
    std::vector<std::size_t> pivots;
    std::size_t prow = 0;
    for (std::size_t col = 0; col < cols && prow < rows; ++col) {
        std::size_t sel = prow;
        bool found = false;
        for (std::size_t r = prow; r < rows; ++r) {
            if (!a[r][col].is_zero()) {
                sel = r;
                found = true;
                break;
            }
        }
        if (!found) {
            continue;  // free column
        }
        std::swap(a[prow], a[sel]);
        const Rational pivot = a[prow][col];
        for (std::size_t j = 0; j < cols; ++j) {
            auto q = a[prow][j].divide(pivot);
            if (!q) {
                return make_error<std::vector<std::size_t>>(q.error());
            }
            a[prow][j] = *q;
        }
        for (std::size_t r = 0; r < rows; ++r) {
            if (r == prow) {
                continue;
            }
            const Rational factor = a[r][col];
            if (factor.is_zero()) {
                continue;
            }
            for (std::size_t j = 0; j < cols; ++j) {
                auto prod = a[prow][j].multiply(factor);
                if (!prod) {
                    return make_error<std::vector<std::size_t>>(prod.error());
                }
                auto diff = a[r][j].subtract(*prod);
                if (!diff) {
                    return make_error<std::vector<std::size_t>>(diff.error());
                }
                a[r][j] = *diff;
            }
        }
        pivots.push_back(col);
        ++prow;
    }
    return pivots;
}

// Right null space of M (m x n): a basis of { v in Q^n : M v = 0 }, returned as rows (each a
// length-n vector). Computed from the RREF: every non-pivot (free) column yields one basis
// vector by setting that free coordinate to 1 and back-substituting the pivots.
[[nodiscard]] auto right_null_space_rows(const Matrix& m) -> Result<RatRows> {
    const std::size_t n = m.cols();
    RatRows a = mat_to_rows(m);
    auto pivots = rref(a, n);
    if (!pivots) {
        return make_error<RatRows>(pivots.error());
    }
    std::vector<bool> is_pivot(n, false);
    for (auto c : *pivots) {
        is_pivot[c] = true;
    }
    RatRows basis;
    for (std::size_t fc = 0; fc < n; ++fc) {
        if (is_pivot[fc]) {
            continue;
        }
        std::vector<Rational> v(n, Rational::from_int(0));
        v[fc] = Rational::from_int(1);
        for (std::size_t i = 0; i < pivots->size(); ++i) {
            auto neg = a[i][fc].negate();  // pivot var = -(free-column entry)
            if (!neg) {
                return make_error<RatRows>(neg.error());
            }
            v[(*pivots)[i]] = *neg;
        }
        basis.push_back(std::move(v));
    }
    return basis;
}

// Left null space of a square E (n x n): rows w (length n) with w E = 0, i.e. the right null
// space of E^T. Empty exactly when E is invertible.
[[nodiscard]] auto left_null_space_rows(const Matrix& e) -> Result<RatRows> {
    auto et = e.transpose();
    if (!et) {
        return make_error<RatRows>(et.error());
    }
    return right_null_space_rows(*et);
}

// Given the null-space rows W (d x n, independent) select r = n - d unit rows e_i that
// complete W to a nonsingular n x n matrix [rows...; W]. Returns the chosen coordinate indices.
[[nodiscard]] auto complete_with_units(const RatRows& w, std::size_t n) -> std::vector<std::size_t> {
    RatRows basis = w;  // start from the null-space rows
    std::vector<std::size_t> chosen;
    const std::size_t need = n - w.size();
    for (std::size_t i = 0; i < n && chosen.size() < need; ++i) {
        std::vector<Rational> unit(n, Rational::from_int(0));
        unit[i] = Rational::from_int(1);
        basis.push_back(std::move(unit));
        if (static_cast<std::size_t>(ratrows_rank(basis)) == basis.size()) {
            chosen.push_back(i);  // independent: keep it
        } else {
            basis.pop_back();  // dependent: drop it
        }
    }
    return chosen;
}

// Negate every component of a series vector.
[[nodiscard]] auto neg_vec(const SeriesVec& v) -> Result<SeriesVec> {
    SeriesVec out;
    out.reserve(v.size());
    const Rational minus_one = Rational::from_int(-1);
    for (const auto& s : v) {
        auto n = s.scale(minus_one);
        if (!n) {
            return make_error<SeriesVec>(n.error());
        }
        out.push_back(std::move(*n));
    }
    return out;
}

// Differentiate every component of a series vector (d/dt in the truncated ring).
[[nodiscard]] auto deriv_vec(const SeriesVec& v) -> Result<SeriesVec> {
    SeriesVec out;
    out.reserve(v.size());
    for (const auto& s : v) {
        auto d = s.derivative();
        if (!d) {
            return make_error<SeriesVec>(d.error());
        }
        out.push_back(std::move(*d));
    }
    return out;
}

// The reduced (index-1 / ODE) form produced by the shuffle algorithm, plus the accumulated
// hidden constraints Gc x + gc = 0 that consistent initial data must satisfy.
struct Reduction {
    Matrix e;       // reduced pencil E*, guaranteed INVERTIBLE
    Matrix a;       // reduced pencil A*
    SeriesVec f;    // reduced forcing f* (order-`order` series)
    std::size_t index{0};
    RatRows gc_mat;   // stacked constraint matrix rows (k x n)
    SeriesVec gc_frc;  // stacked constraint forcings (k), order-`order` series
    std::size_t n{0};
};

// Run the shuffle algorithm on E x' = A x + f (f already retruncated to `order`), reducing to
// an invertible pencil and counting the differentiation index. Fails with domain_error if the
// pencil is non-regular (still singular after n passes) and propagates every exact error.
[[nodiscard]] auto shuffle_reduce(Matrix e, Matrix a, SeriesVec f, std::size_t order)
    -> Result<Reduction> {
    const std::size_t n = e.rows();
    RatRows gc_mat;
    SeriesVec gc_frc;
    std::size_t index = 0;
    while (true) {
        auto w = left_null_space_rows(e);
        if (!w) {
            return make_error<Reduction>(w.error());
        }
        if (w->empty()) {
            break;  // E invertible: reduction complete
        }
        if (index == n) {
            return make_error<Reduction>(MathError::domain_error);  // non-regular pencil
        }
        auto wmat = Matrix::from_rows(*w);  // d x n (d >= 1 here)
        if (!wmat) {
            return make_error<Reduction>(wmat.error());
        }
        // Algebraic constraint block A2 x + (W f) = 0 with A2 = W A.
        auto a2 = wmat->multiply(a);
        if (!a2) {
            return make_error<Reduction>(a2.error());
        }
        auto wf = mat_series_vec(*wmat, f, order);
        if (!wf) {
            return make_error<Reduction>(wf.error());
        }
        // Accumulate the (undifferentiated) hidden constraints on the trajectory. wf is COPIED
        // (not moved) because it is differentiated again below to build the new forcing.
        for (auto& row : mat_to_rows(*a2)) {
            gc_mat.push_back(std::move(row));
        }
        for (const auto& s : *wf) {
            gc_frc.push_back(s);
        }
        // Differential rows to keep: r unit selections completing W to full rank.
        const std::vector<std::size_t> keep = complete_with_units(*w, n);
        const RatRows e_rows = mat_to_rows(e);
        const RatRows a_rows = mat_to_rows(a);
        const RatRows a2_rows = mat_to_rows(*a2);
        // New pencil: E_new = [E rows kept ; A2],  A_new = [A rows kept ; 0].
        RatRows e_new;
        RatRows a_new;
        e_new.reserve(n);
        a_new.reserve(n);
        SeriesVec f_new;
        f_new.reserve(n);
        for (const std::size_t i : keep) {
            e_new.push_back(e_rows[i]);
            a_new.push_back(a_rows[i]);
            f_new.push_back(f[i]);
        }
        for (const auto& row : a2_rows) {
            e_new.push_back(row);
            a_new.push_back(std::vector<Rational>(n, Rational::from_int(0)));  // no x-term
        }
        // Differentiated constraint forcing: -(W f)'.
        auto wf_deriv = deriv_vec(*wf);
        if (!wf_deriv) {
            return make_error<Reduction>(wf_deriv.error());
        }
        auto neg_wf_deriv = neg_vec(*wf_deriv);
        if (!neg_wf_deriv) {
            return make_error<Reduction>(neg_wf_deriv.error());
        }
        for (auto& s : *neg_wf_deriv) {
            f_new.push_back(std::move(s));
        }
        auto e_next = Matrix::from_rows(std::move(e_new));
        if (!e_next) {
            return make_error<Reduction>(e_next.error());
        }
        auto a_next = Matrix::from_rows(std::move(a_new));
        if (!a_next) {
            return make_error<Reduction>(a_next.error());
        }
        e = std::move(*e_next);
        a = std::move(*a_next);
        f = std::move(f_new);
        ++index;
    }
    return Reduction{std::move(e), std::move(a), std::move(f), index,
                     std::move(gc_mat), std::move(gc_frc), n};
}

// Consistency residual of x0 against the accumulated constraints: entry k is
// sum_j gc_mat[k][j] x0[j] + gc_frc[k](0). x0 is consistent iff every entry is zero.
[[nodiscard]] auto constraint_residual(const RatRows& gc_mat, const SeriesVec& gc_frc,
                                       const std::vector<Rational>& x0)
    -> Result<std::vector<Rational>> {
    std::vector<Rational> res;
    res.reserve(gc_mat.size());
    for (std::size_t k = 0; k < gc_mat.size(); ++k) {
        Rational acc = gc_frc[k].coefficient(0);  // forcing value at t = 0
        for (std::size_t j = 0; j < x0.size(); ++j) {
            auto term = gc_mat[k][j].multiply(x0[j]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto sum = acc.add(*term);
            if (!sum) {
                return make_error<std::vector<Rational>>(sum.error());
            }
            acc = *sum;
        }
        res.push_back(acc);
    }
    return res;
}

[[nodiscard]] auto all_zero(const std::vector<Rational>& v) -> bool {
    return std::ranges::all_of(v, [](const Rational& r) { return r.is_zero(); });
}

// Orthogonally project x0 (exactly over Q) onto { x : Gc x = h } with h = -gc(0), where Gc is
// the accumulated constraint matrix. Reduces Gc to a maximal independent row set Gr (so
// Gr Gr^T is invertible) and returns x0 - Gr^T (Gr Gr^T)^{-1} (Gr x0 - hr).
[[nodiscard]] auto project_onto_manifold(const RatRows& gc_mat, const SeriesVec& gc_frc,
                                         const std::vector<Rational>& x0, std::size_t n)
    -> Result<std::vector<Rational>> {
    if (gc_mat.empty()) {
        return x0;  // no constraints: every x0 is consistent
    }
    // Select a maximal set of independent constraint rows.
    RatRows basis;
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < gc_mat.size(); ++i) {
        basis.push_back(gc_mat[i]);
        if (static_cast<std::size_t>(ratrows_rank(basis)) == basis.size()) {
            keep.push_back(i);
        } else {
            basis.pop_back();
        }
    }
    if (keep.empty()) {
        return x0;  // all constraints trivial (zero rows)
    }
    RatRows gr_rows;
    gr_rows.reserve(keep.size());
    std::vector<Rational> hr;  // h = -gc(0) restricted to kept rows
    hr.reserve(keep.size());
    for (const std::size_t i : keep) {
        gr_rows.push_back(gc_mat[i]);
        auto neg = gc_frc[i].coefficient(0).negate();
        if (!neg) {
            return make_error<std::vector<Rational>>(neg.error());
        }
        hr.push_back(*neg);
    }
    auto gr = Matrix::from_rows(std::move(gr_rows));  // k x n, full row rank
    if (!gr) {
        return make_error<std::vector<Rational>>(gr.error());
    }
    auto grt = gr->transpose();  // n x k
    if (!grt) {
        return make_error<std::vector<Rational>>(grt.error());
    }
    auto ggt = gr->multiply(*grt);  // k x k, invertible (Gr full row rank)
    if (!ggt) {
        return make_error<std::vector<Rational>>(ggt.error());
    }
    // residual = Gr x0 - hr, as a k x 1 matrix.
    const std::size_t k = keep.size();
    std::vector<std::vector<Rational>> res_rows;
    res_rows.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        Rational acc;  // 0/1
        for (std::size_t j = 0; j < n; ++j) {
            auto term = gr->at(i, j).multiply(x0[j]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto sum = acc.add(*term);
            if (!sum) {
                return make_error<std::vector<Rational>>(sum.error());
            }
            acc = *sum;
        }
        auto diff = acc.subtract(hr[i]);
        if (!diff) {
            return make_error<std::vector<Rational>>(diff.error());
        }
        res_rows.push_back({*diff});
    }
    auto res = Matrix::from_rows(std::move(res_rows));  // k x 1
    if (!res) {
        return make_error<std::vector<Rational>>(res.error());
    }
    auto lambda = ggt->solve(*res);  // (Gr Gr^T) lambda = residual, k x 1
    if (!lambda) {
        return make_error<std::vector<Rational>>(lambda.error());
    }
    auto corr = grt->multiply(*lambda);  // Gr^T lambda, n x 1
    if (!corr) {
        return make_error<std::vector<Rational>>(corr.error());
    }
    std::vector<Rational> out;
    out.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        auto d = x0[j].subtract(corr->at(j, 0));
        if (!d) {
            return make_error<std::vector<Rational>>(d.error());
        }
        out.push_back(*d);
    }
    return out;
}

// Validate the general-DAE shapes and retruncate the forcing to `order`. On success returns
// the working forcing vector.
[[nodiscard]] auto prepare_general_dae(const Matrix& e, const Matrix& a, const SeriesVec& f,
                                       const std::vector<Rational>& x0, std::size_t order)
    -> Result<SeriesVec> {
    if (order == 0 || !e.is_square() || !a.is_square() || e.rows() != a.rows()) {
        return make_error<SeriesVec>(MathError::domain_error);
    }
    const std::size_t n = e.rows();
    if (n == 0 || f.size() != n || x0.size() != n) {
        return make_error<SeriesVec>(MathError::domain_error);
    }
    return retruncate_vec(f, order);
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

// ===========================================================================
// Higher-index linear constant-coefficient DAE implementation.
// ===========================================================================

auto linear_dae_index(const Matrix& E, const Matrix& A) -> Result<std::size_t> {
    if (!E.is_square() || !A.is_square() || E.rows() != A.rows()) {
        return make_error<std::size_t>(MathError::domain_error);
    }
    const std::size_t n = E.rows();
    if (n == 0) {
        return make_error<std::size_t>(MathError::domain_error);
    }
    // The index is forcing-independent; reduce with a zero forcing (order 1 suffices).
    SeriesVec f;
    f.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto z = PowerSeries::zero(1);
        if (!z) {
            return make_error<std::size_t>(z.error());
        }
        f.push_back(std::move(*z));
    }
    auto red = shuffle_reduce(E, A, std::move(f), 1);
    if (!red) {
        return make_error<std::size_t>(red.error());
    }
    return red->index;
}

auto linear_dae_is_consistent(const Matrix& E, const Matrix& A, const SeriesVec& f,
                              const std::vector<Rational>& x0, std::size_t order) -> Result<bool> {
    auto fr = prepare_general_dae(E, A, f, x0, order);
    if (!fr) {
        return make_error<bool>(fr.error());
    }
    auto red = shuffle_reduce(E, A, std::move(*fr), order);
    if (!red) {
        return make_error<bool>(red.error());
    }
    auto res = constraint_residual(red->gc_mat, red->gc_frc, x0);
    if (!res) {
        return make_error<bool>(res.error());
    }
    return all_zero(*res);
}

auto project_to_consistent(const Matrix& E, const Matrix& A, const SeriesVec& f,
                           const std::vector<Rational>& x0, std::size_t order)
    -> Result<std::vector<Rational>> {
    auto fr = prepare_general_dae(E, A, f, x0, order);
    if (!fr) {
        return make_error<std::vector<Rational>>(fr.error());
    }
    auto red = shuffle_reduce(E, A, std::move(*fr), order);
    if (!red) {
        return make_error<std::vector<Rational>>(red.error());
    }
    return project_onto_manifold(red->gc_mat, red->gc_frc, x0, red->n);
}

auto solve_linear_dae(const Matrix& E, const Matrix& A, const SeriesVec& f,
                      const std::vector<Rational>& x0, std::size_t order, ConsistencyPolicy policy)
    -> Result<LinearDaeSolution> {
    auto fr = prepare_general_dae(E, A, f, x0, order);
    if (!fr) {
        return make_error<LinearDaeSolution>(fr.error());
    }
    const std::size_t n = E.rows();

    // Index reduction: shuffle to an invertible pencil, collecting hidden constraints.
    auto red = shuffle_reduce(E, A, std::move(*fr), order);
    if (!red) {
        return make_error<LinearDaeSolution>(red.error());
    }

    // Consistency of the supplied x0 against the accumulated constraints at t = 0.
    auto res = constraint_residual(red->gc_mat, red->gc_frc, x0);
    if (!res) {
        return make_error<LinearDaeSolution>(res.error());
    }
    const bool consistent = all_zero(*res);

    // Resolve the initial vector according to policy.
    std::vector<Rational> x0_used = x0;
    if (!consistent) {
        switch (policy) {
            case ConsistencyPolicy::require:
                return make_error<LinearDaeSolution>(MathError::domain_error);
            case ConsistencyPolicy::project: {
                auto proj = project_onto_manifold(red->gc_mat, red->gc_frc, x0, n);
                if (!proj) {
                    return make_error<LinearDaeSolution>(proj.error());
                }
                x0_used = std::move(*proj);
                break;
            }
            case ConsistencyPolicy::underlying:
                break;  // solve the underlying ODE from x0 as given
        }
    }

    // Reduce the invertible-pencil DAE onto the ODE: x' = E*^{-1} A* x + E*^{-1} f*.
    auto einv = red->e.inverse();
    if (!einv) {
        return make_error<LinearDaeSolution>(einv.error());
    }
    auto M = einv->multiply(red->a);
    if (!M) {
        return make_error<LinearDaeSolution>(M.error());
    }
    auto rforce = mat_series_vec(*einv, red->f, order);
    if (!rforce) {
        return make_error<LinearDaeSolution>(rforce.error());
    }

    Matrix Mmat = std::move(*M);
    SeriesVec rvec = std::move(*rforce);
    SystemOperator field = [Mmat, rvec, order](const SeriesVec& u) -> Result<SeriesVec> {
        auto mx = mat_series_vec(Mmat, u, order);
        if (!mx) {
            return make_error<SeriesVec>(mx.error());
        }
        return vec_add(*mx, rvec);
    };
    auto xsol = solve_first_order_system(std::move(field), x0_used, order);
    if (!xsol) {
        return make_error<LinearDaeSolution>(xsol.error());
    }

    return LinearDaeSolution{std::move(*xsol), red->index, consistent};
}

auto solve_semiexplicit_dae(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D,
                            const SeriesVec& p, const SeriesVec& q, const std::vector<Rational>& x0,
                            const std::vector<Rational>& y0, std::size_t order,
                            ConsistencyPolicy policy) -> Result<SemiExplicitDaeSolution> {
    if (order == 0 || !A.is_square() || !D.is_square()) {
        return make_error<SemiExplicitDaeSolution>(MathError::domain_error);
    }
    const std::size_t nd = A.rows();
    const std::size_t na = D.rows();
    if (B.rows() != nd || B.cols() != na || C.rows() != na || C.cols() != nd) {
        return make_error<SemiExplicitDaeSolution>(MathError::domain_error);
    }
    if (p.size() != nd || q.size() != na || x0.size() != nd || y0.size() != na) {
        return make_error<SemiExplicitDaeSolution>(MathError::domain_error);
    }
    const std::size_t n = nd + na;
    const Rational zero = Rational::from_int(0);
    const Rational one = Rational::from_int(1);

    // E = diag(I_nd, 0_na).
    RatRows e_rows(n, std::vector<Rational>(n, zero));
    for (std::size_t i = 0; i < nd; ++i) {
        e_rows[i][i] = one;
    }
    auto E = Matrix::from_rows(std::move(e_rows));
    if (!E) {
        return make_error<SemiExplicitDaeSolution>(E.error());
    }
    // M = [[A, B], [C, D]].
    RatRows m_rows(n, std::vector<Rational>(n, zero));
    for (std::size_t i = 0; i < nd; ++i) {
        for (std::size_t j = 0; j < nd; ++j) {
            m_rows[i][j] = A.at(i, j);
        }
        for (std::size_t j = 0; j < na; ++j) {
            m_rows[i][nd + j] = B.at(i, j);
        }
    }
    for (std::size_t i = 0; i < na; ++i) {
        for (std::size_t j = 0; j < nd; ++j) {
            m_rows[nd + i][j] = C.at(i, j);
        }
        for (std::size_t j = 0; j < na; ++j) {
            m_rows[nd + i][nd + j] = D.at(i, j);
        }
    }
    auto M = Matrix::from_rows(std::move(m_rows));
    if (!M) {
        return make_error<SemiExplicitDaeSolution>(M.error());
    }
    // f = [p; q], z0 = [x0; y0].
    SeriesVec f;
    f.reserve(n);
    for (const auto& s : p) {
        f.push_back(s);
    }
    for (const auto& s : q) {
        f.push_back(s);
    }
    std::vector<Rational> z0;
    z0.reserve(n);
    z0.insert(z0.end(), x0.begin(), x0.end());
    z0.insert(z0.end(), y0.begin(), y0.end());

    auto sol = solve_linear_dae(*E, *M, f, z0, order, policy);
    if (!sol) {
        return make_error<SemiExplicitDaeSolution>(sol.error());
    }
    // Split z back into x (first nd) and y (last na).
    SemiExplicitDaeSolution out;
    out.index = sol->index;
    out.consistent = sol->consistent;
    out.x.reserve(nd);
    out.y.reserve(na);
    for (std::size_t i = 0; i < nd; ++i) {
        out.x.push_back(std::move(sol->x[i]));
    }
    for (std::size_t i = 0; i < na; ++i) {
        out.y.push_back(std::move(sol->x[nd + i]));
    }
    return out;
}

}  // namespace nimblecas
