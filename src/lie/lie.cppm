// NimbleCAS matrix Lie algebras & Lie transforms over the rationals (ROADMAP 7.x).
// @author Olumuyiwa Oluwasanmi
//
// A matrix Lie algebra is a vector space of square matrices over Q closed under the Lie
// bracket [A,B] = AB - BA. This module supplies the exact-over-Q toolkit for such algebras:
// the bracket itself, the structure constants c^k_ij of a chosen basis, the adjoint
// representation ad_X and the Killing form K(X,Y) = trace(ad_X ad_Y), the exponential map
// (delegated to nimblecas.matexp), and truncated Lie series / adjoint-action transforms.
//
// Everything rides the Result railway (Rule 32): dimension violations and a basis that is
// not closed under the bracket surface as MathError::domain_error, and any Rational int64
// overflow is propagated (never silently wrapped).
//
// -----------------------------------------------------------------------------
// HONESTY BOUNDARY — what is exact and what is a truncation.
// -----------------------------------------------------------------------------
//   * EXACT over Q (no approximation whatsoever):
//       - lie_bracket(A,B) = AB - BA. Bilinearity, antisymmetry [A,B] = -[B,A], and the
//         Jacobi identity [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0 hold identically.
//       - structure_constants(basis): the unique c^k_ij with [X_i,X_j] = Sum_k c^k_ij X_k,
//         found by an exact rational linear solve (each bracket expanded in the basis). If a
//         bracket leaves the span, the algebra is not closed and this returns domain_error.
//       - adjoint_matrix(X, basis): ad_X as an exact n x n rational matrix in the basis
//         (column j is the basis-expansion of [X, X_j]).
//       - killing_form(X,Y,basis) = trace(ad_X ad_Y): an exact Rational.
//       - lie_series / adjoint_action_series: the TRUNCATED sum Sum_{k=0}^{order} (t^k/k!)
//         ad_L^k(f), computed with exact iterated brackets. Every term it keeps is exact;
//         it is the whole (infinite) transform ONLY in the limit order -> infinity, or, for
//         a nilpotent ad_L, once `order` reaches the nilpotency index of ad_L (the omitted
//         tail is then identically zero). Otherwise it is an exact PARTIAL sum, honestly
//         truncated — never a claim of the full Ad_{exp(L)}.
//   * INHERITED contract (exact-iff-nilpotent, else approximate):
//       - exponential_map(A, terms) merely forwards to nimblecas.matexp's matrix_exp_taylor:
//         the true e^A is generally transcendental and NOT representable over Q. The returned
//         truncated Taylor polynomial equals e^A EXACTLY iff A is nilpotent and `terms` is at
//         least A's nilpotency index; for any other A it is an exact rational approximation.
//         See the nimblecas.matexp header for the full statement.

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.lie;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational
import nimblecas.matrix;    // Matrix (exact dense linear algebra over Q)
import nimblecas.matexp;    // matrix_exp_taylor — the exponential map (honest contract)

export namespace nimblecas {

// ---------------------------------------------------------------------------
// The Lie bracket [A,B] = A*B - B*A.
// ---------------------------------------------------------------------------
// A and B must be square and of the same order, else domain_error. The result is exact
// over Q; overflow in an entry is propagated. This is also the adjoint action ad_A(B):
// ad_A(B) = [A,B].
[[nodiscard]] auto lie_bracket(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// StructureConstants — the tensor c^k_ij with [X_i,X_j] = Sum_k c^k_ij X_k.
// ---------------------------------------------------------------------------
// dimension() is the number of basis elements n; the constants are indexed i,j,k in [0,n).
// Antisymmetry c^k_ij = -c^k_ji holds by construction of the bracket.
class StructureConstants {
public:
    StructureConstants() = default;
    StructureConstants(std::size_t dim, std::vector<Rational> data)
        : dim_(dim), data_(std::move(data)) {}

    [[nodiscard]] auto dimension() const noexcept -> std::size_t { return dim_; }

    // c^k_ij : the coefficient of X_k in the expansion of [X_i, X_j]. Indices asserted
    // in-range (callers hold i,j,k below dimension()).
    [[nodiscard]] auto at(std::size_t i, std::size_t j, std::size_t k) const -> const Rational& {
        assert(i < dim_ && j < dim_ && k < dim_ && "StructureConstants::at out of range");
        return data_[(i * dim_ + j) * dim_ + k];
    }

private:
    std::size_t dim_{0};
    std::vector<Rational> data_{};  // flat n^3, index ((i*n)+j)*n + k
};

// Compute the structure constants of a basis {X_0..X_{n-1}} of square matrices over Q.
// The basis must be linearly independent (else the linear solve is singular -> domain_error)
// and CLOSED under the bracket: if some [X_i,X_j] is not in the span of the basis, the
// algebra is not closed and this returns domain_error. Overflow is propagated.
[[nodiscard]] auto structure_constants(const std::vector<Matrix>& basis)
    -> Result<StructureConstants>;

// ---------------------------------------------------------------------------
// Adjoint representation and Killing form.
// ---------------------------------------------------------------------------
// ad_X as an exact n x n rational matrix in the given basis: entry (k,j) is the coefficient
// of X_k in [X, X_j], so (ad_X)(coeffs of Z) = coeffs of [X,Z]. Requires the basis to be
// linearly independent and closed under bracketing with X (else domain_error). X must share
// the basis matrices' order. Overflow is propagated.
[[nodiscard]] auto adjoint_matrix(const Matrix& x, const std::vector<Matrix>& basis)
    -> Result<Matrix>;

// The Killing form K(X,Y) = trace(ad_X ad_Y), an exact Rational computed from the adjoint
// matrices in the basis. Propagates the same domain_error / overflow conditions as
// adjoint_matrix.
[[nodiscard]] auto killing_form(const Matrix& x, const Matrix& y,
                                const std::vector<Matrix>& basis) -> Result<Rational>;

// ---------------------------------------------------------------------------
// Exponential map (delegated to nimblecas.matexp — inherits its honesty contract).
// ---------------------------------------------------------------------------
// exp(A) as the truncated Taylor polynomial Sum_{k=0}^{terms-1} A^k/k!. EXACT e^A iff A is
// nilpotent and terms >= A's nilpotency index; otherwise an exact rational APPROXIMATION of
// the (generally transcendental) e^A. Requires A square and terms >= 1, else domain_error.
[[nodiscard]] auto exponential_map(const Matrix& a, std::int64_t terms) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// Lie series / Lie transform.
// ---------------------------------------------------------------------------
// The truncated Lie series exp(t ad_L) f = Sum_{k=0}^{order} (t^k/k!) ad_L^k(f), where
// ad_L^k(f) is the k-fold nested bracket [L,[L,...[L,f]...]]. Each term is exact over Q
// (iterated brackets, exact Rational t^k/k!); the sum is truncated at `order` and is the
// full transform only in the limit, or once `order` reaches the nilpotency index of ad_L
// (see the module header). L and f must be square and of equal order; order >= 0. Overflow
// is propagated.
[[nodiscard]] auto lie_series(const Matrix& l, const Matrix& f, const Rational& t,
                              std::int64_t order) -> Result<Matrix>;

// The adjoint action Ad_{exp(L)} f = exp(L) f exp(-L) approximated by its Lie series at t=1:
// Sum_{k=0}^{order} ad_L^k(f)/k!. A convenience wrapper over lie_series(l, f, 1, order) with
// the identical truncation honesty.
[[nodiscard]] auto adjoint_action_series(const Matrix& l, const Matrix& f, std::int64_t order)
    -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Require A and B to be square and of the same order; that is the precondition shared by the
// bracket and every algebra element here.
[[nodiscard]] auto square_same_order(const Matrix& a, const Matrix& b) -> bool {
    return a.is_square() && b.is_square() && a.rows() == b.rows();
}

// Express `target` in the basis {X_0..X_{n-1}} of square d x d matrices: find rational
// coefficients (a_0..a_{n-1}) with target = Sum_k a_k X_k, exactly. Each matrix is viewed as
// a d^2 vector; stacking the basis vectors as columns gives an d^2 x n system M a = vec(t).
// It is solved through the (exact) normal equations (M^T M) a = M^T vec(t): M^T M is n x n
// and invertible precisely when the basis is linearly independent, so a singular system
// surfaces as domain_error. The recovered a is then substituted back and M a is checked to
// equal vec(t); a mismatch means `target` is NOT in the span (the algebra is not closed),
// which is likewise reported as domain_error. Every step is overflow-checked.
[[nodiscard]] auto expand_in_basis(const Matrix& target, const std::vector<Matrix>& basis)
    -> Result<std::vector<Rational>> {
    if (!target.is_square()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::size_t d = target.rows();
    const std::size_t n = basis.size();

    // An empty basis spans only the zero matrix.
    if (n == 0) {
        if (target == Matrix::zero(d, d)) {
            return std::vector<Rational>{};
        }
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    for (const Matrix& x : basis) {
        if (!x.is_square() || x.rows() != d) {
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
    }

    const std::size_t rows = d * d;  // vec() length

    // M (rows x n): column k is vec(basis[k]); and the right-hand column vector vec(target).
    std::vector<std::vector<Rational>> m_rows(rows, std::vector<Rational>(n));
    std::vector<std::vector<Rational>> b_rows(rows, std::vector<Rational>(1));
    for (std::size_t e = 0; e < rows; ++e) {
        const std::size_t r = e / d;
        const std::size_t c = e % d;
        for (std::size_t k = 0; k < n; ++k) {
            m_rows[e][k] = basis[k].at(r, c);
        }
        b_rows[e][0] = target.at(r, c);
    }
    auto m = Matrix::from_rows(std::move(m_rows));
    if (!m) {
        return make_error<std::vector<Rational>>(m.error());
    }
    auto b = Matrix::from_rows(std::move(b_rows));
    if (!b) {
        return make_error<std::vector<Rational>>(b.error());
    }

    // Normal equations: (M^T M) a = M^T b.
    auto mt = m->transpose();
    if (!mt) {
        return make_error<std::vector<Rational>>(mt.error());
    }
    auto gram = mt->multiply(*m);  // n x n
    if (!gram) {
        return make_error<std::vector<Rational>>(gram.error());
    }
    auto rhs = mt->multiply(*b);  // n x 1
    if (!rhs) {
        return make_error<std::vector<Rational>>(rhs.error());
    }
    auto coeffs = gram->solve(*rhs);  // singular Gram (dependent basis) -> domain_error
    if (!coeffs) {
        return make_error<std::vector<Rational>>(coeffs.error());
    }

    // Verify the solution actually reconstructs the target (target is in the span).
    auto recon = m->multiply(*coeffs);  // rows x 1
    if (!recon) {
        return make_error<std::vector<Rational>>(recon.error());
    }
    if (!(*recon == *b)) {
        return make_error<std::vector<Rational>>(MathError::domain_error);  // not closed
    }

    std::vector<Rational> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        out[k] = coeffs->at(k, 0);
    }
    return out;
}

}  // namespace

// --- Lie bracket ------------------------------------------------------------

auto lie_bracket(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    if (!square_same_order(a, b)) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto ab = a.multiply(b);
    if (!ab) {
        return make_error<Matrix>(ab.error());
    }
    auto ba = b.multiply(a);
    if (!ba) {
        return make_error<Matrix>(ba.error());
    }
    return ab->subtract(*ba);  // AB - BA
}

// --- structure constants ----------------------------------------------------

auto structure_constants(const std::vector<Matrix>& basis) -> Result<StructureConstants> {
    const std::size_t n = basis.size();
    std::vector<Rational> data(n * n * n);  // Rational{} default is 0/1
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto bracket = lie_bracket(basis[i], basis[j]);
            if (!bracket) {
                return make_error<StructureConstants>(bracket.error());
            }
            auto coeffs = expand_in_basis(*bracket, basis);  // domain_error if not closed
            if (!coeffs) {
                return make_error<StructureConstants>(coeffs.error());
            }
            for (std::size_t k = 0; k < n; ++k) {
                data[(i * n + j) * n + k] = (*coeffs)[k];
            }
        }
    }
    return StructureConstants{n, std::move(data)};
}

// --- adjoint representation / Killing form -----------------------------------

auto adjoint_matrix(const Matrix& x, const std::vector<Matrix>& basis) -> Result<Matrix> {
    const std::size_t n = basis.size();
    // Row k, column j holds the coefficient of X_k in [X, X_j].
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n));
    for (std::size_t j = 0; j < n; ++j) {
        auto bracket = lie_bracket(x, basis[j]);
        if (!bracket) {
            return make_error<Matrix>(bracket.error());
        }
        auto coeffs = expand_in_basis(*bracket, basis);
        if (!coeffs) {
            return make_error<Matrix>(coeffs.error());
        }
        for (std::size_t k = 0; k < n; ++k) {
            rows[k][j] = (*coeffs)[k];
        }
    }
    return Matrix::from_rows(std::move(rows));
}

auto killing_form(const Matrix& x, const Matrix& y, const std::vector<Matrix>& basis)
    -> Result<Rational> {
    auto ad_x = adjoint_matrix(x, basis);
    if (!ad_x) {
        return make_error<Rational>(ad_x.error());
    }
    auto ad_y = adjoint_matrix(y, basis);
    if (!ad_y) {
        return make_error<Rational>(ad_y.error());
    }
    auto prod = ad_x->multiply(*ad_y);
    if (!prod) {
        return make_error<Rational>(prod.error());
    }
    return prod->trace();  // trace(ad_X ad_Y)
}

// --- exponential map (delegated) --------------------------------------------

auto exponential_map(const Matrix& a, std::int64_t terms) -> Result<Matrix> {
    // Honest by delegation: matrix_exp_taylor is exact for e^A iff A is nilpotent and
    // terms >= its nilpotency index; otherwise it is a truncated rational approximation.
    return matrix_exp_taylor(a, terms);
}

// --- Lie series / adjoint action --------------------------------------------

auto lie_series(const Matrix& l, const Matrix& f, const Rational& t, std::int64_t order)
    -> Result<Matrix> {
    if (!square_same_order(l, f) || order < 0) {
        return make_error<Matrix>(MathError::domain_error);
    }
    // result = Sum_{k=0}^{order} coeff_k * bracket_k, with bracket_0 = f (ad_L^0 f) and
    // coeff_0 = t^0/0! = 1. Each step advances bracket_k = [L, bracket_{k-1}] = ad_L^k(f)
    // and coeff_k = coeff_{k-1} * t / k, so no large factorial or power is ever formed.
    Matrix result = f;
    Matrix bracket = f;
    Rational coeff = Rational::from_int(1);
    for (std::int64_t k = 1; k <= order; ++k) {
        auto next_bracket = lie_bracket(l, bracket);
        if (!next_bracket) {
            return make_error<Matrix>(next_bracket.error());
        }
        bracket = *next_bracket;

        auto ct = coeff.multiply(t);
        if (!ct) {
            return make_error<Matrix>(ct.error());
        }
        auto ck = ct->divide(Rational::from_int(k));  // k >= 1, never a zero divisor
        if (!ck) {
            return make_error<Matrix>(ck.error());
        }
        coeff = *ck;

        auto term = bracket.scale(coeff);
        if (!term) {
            return make_error<Matrix>(term.error());
        }
        auto sum = result.add(*term);
        if (!sum) {
            return make_error<Matrix>(sum.error());
        }
        result = *sum;
    }
    return result;
}

auto adjoint_action_series(const Matrix& l, const Matrix& f, std::int64_t order)
    -> Result<Matrix> {
    return lie_series(l, f, Rational::from_int(1), order);  // Ad_{exp L} f at t = 1
}

}  // namespace nimblecas
