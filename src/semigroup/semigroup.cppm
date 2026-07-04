// NimbleCAS operator theory & C0-semigroups (finite-dimensional realization).
// @author Olumuyiwa Oluwasanmi
//
// Functional analysis and one-parameter operator semigroups, realized in the tractable
// FINITE-DIMENSIONAL (matrix-operator) setting over the exact rationals Q. A square rational
// matrix A is read as a bounded linear operator on Q^n; the associated C0-semigroup is
// T(t) = e^{tA}, the flow of the abstract Cauchy problem du/dt = A u, u(0) = u0.
//
// WHAT IS EXACT AND WHAT IS NOT.
//   * EXACT over Q (no rounding whatsoever):
//       - the resolvent R(lambda, A) = (lambda*I - A)^{-1} (via Matrix::inverse over Q),
//       - the RATIONAL part of the spectrum (rational eigenvalues + multiplicities),
//       - the induced operator 1-norm and inf-norm (max abs column / row sums),
//       - the adjoint (transpose for a real rational operator),
//       - the Sylvester solve A X + X B = C and its Lyapunov special case B = A^T,
//       - the dissipativity verdict (A + A^T negative semidefinite) and the Hurwitz verdict
//         (spectrum in the open left half-plane) -- both decided from characteristic-
//         polynomial coefficient signs, so they settle irrational/complex spectra too.
//   * INHERITED from nimblecas.matexp (exact iff nilpotent, else an exact-RATIONAL
//     truncation of the transcendental e^{tA}):
//       - the semigroup T(t) = e^{tA} and everything built on it: the Cauchy solution
//         T(t)u0, the semigroup-property check T(s+t) = T(s)T(t), variation of constants,
//         and the abstract-PDE helper. For a nilpotent generator with enough Taylor terms
//         these are genuinely exact; otherwise they are the honest rational truncation and
//         are labelled as such -- never a claim of the transcendental truth.
//   * SPECTRUM HONESTY: irrational/complex eigenvalues are NOT extracted (that needs
//     algebraic-number support, out of scope). Spectrum results report whether the rational
//     eigenvalues account for the whole spectrum ("fully extracted") or not.
//
// TWO IDENTITIES DOCUMENTED HERE (stated, used to structure the API):
//   * Generator / Cauchy problem: u(t) = T(t) u0 solves u' = A u, u(0) = u0, and
//     d/dt T(t)|_{t=0} = A (A is the infinitesimal generator).
//   * Resolvent as Laplace transform: for Re(lambda) greater than the spectral abscissa,
//         R(lambda, A) = (lambda*I - A)^{-1} = \int_0^\infty e^{-lambda t} T(t) dt,
//     connecting the resolvent (exact over Q) to the semigroup (transcendental in general).
//
// SCOPE. This is the finite-dimensional / matrix realization of semigroup theory. Genuine
// infinite-dimensional functional analysis -- unbounded operators, Banach-space C0-
// semigroups, domains and cores, spectral theory of differential operators -- is OUT OF
// SCOPE and is not claimed. The abstract-PDE helper only consumes an already-discretized
// (finite-dimensional) generator matrix, e.g. a spatial discretization from nimblecas.pde.

export module nimblecas.semigroup;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matexp;
import nimblecas.eigen;
import nimblecas.dynamics;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Spectrum report (rational slice, honest about the rest).
// ---------------------------------------------------------------------------
// `rational_values` pairs each rational eigenvalue with its algebraic multiplicity;
// `rational_count` is the sum of those multiplicities and `dimension` is n. When
// `fully_extracted` is true (rational_count == dimension) the rational eigenvalues are the
// WHOLE spectrum; otherwise part of the spectrum is irrational/complex and not represented.
struct Spectrum {
    std::vector<std::pair<Rational, std::int64_t>> rational_values;
    std::int64_t rational_count{0};
    std::int64_t dimension{0};
    bool fully_extracted{false};
};

// Spectral radius rho(A) = max |eigenvalue|. When `exact` is true (spectrum fully rational)
// `value` is the EXACT spectral radius. When `exact` is false part of the spectrum is not
// over Q, so `value` is instead a guaranteed rational UPPER BOUND on rho(A) -- the smaller of
// the induced 1-norm and inf-norm (rho(A) <= ||A|| for any submultiplicative norm) -- never
// a claim of the exact radius.
struct SpectralRadius {
    Rational value;
    bool exact{false};
};

// ---------------------------------------------------------------------------
// Operator / functional-analysis tools (all EXACT over Q).
// ---------------------------------------------------------------------------

// The resolvent R(lambda, A) = (lambda*I - A)^{-1}, computed EXACTLY over Q. Requires A
// square (else domain_error). At a spectral value of A -- i.e. when lambda is an eigenvalue,
// so (lambda*I - A) is singular -- the inverse does not exist and this surfaces as
// domain_error (propagated from Matrix::inverse). Overflow is propagated.
[[nodiscard]] auto resolvent(const Matrix& a, const Rational& lambda) -> Result<Matrix>;

// The rational slice of the spectrum of A (see Spectrum). Requires A square (else
// domain_error). Exact over Q; irrational/complex eigenvalues are reported as not fully
// extracted rather than approximated.
[[nodiscard]] auto spectrum(const Matrix& a) -> Result<Spectrum>;

// The spectral radius (see SpectralRadius): exact max |rational eigenvalue| when the spectrum
// is fully rational, otherwise a rational upper bound. Requires A square (else domain_error).
[[nodiscard]] auto spectral_radius(const Matrix& a) -> Result<SpectralRadius>;

// The adjoint operator. For a real rational operator the adjoint is the TRANSPOSE, which is
// what this returns exactly. (For a complex operator the adjoint is the conjugate-transpose;
// complex matrices live in nimblecas.cmatrix and are out of scope for this real-Q module.)
[[nodiscard]] auto adjoint(const Matrix& a) -> Result<Matrix>;

// The induced operator 1-norm ||A||_1 = max over columns j of sum_i |A(i,j)| (max absolute
// column sum), exact over Q. Defined for any shape; the empty matrix has norm 0. Overflow
// is propagated.
[[nodiscard]] auto operator_norm_1(const Matrix& a) -> Result<Rational>;

// The induced operator inf-norm ||A||_inf = max over rows i of sum_j |A(i,j)| (max absolute
// row sum), exact over Q. Defined for any shape; the empty matrix has norm 0. Overflow is
// propagated.
[[nodiscard]] auto operator_norm_inf(const Matrix& a) -> Result<Rational>;

// ---------------------------------------------------------------------------
// C0-semigroup generated by A: T(t) = e^{tA}.
// ---------------------------------------------------------------------------

// The semigroup operator T(t) = e^{tA}, computed as the truncated Taylor series of (t*A)
// with `terms` terms (via nimblecas.matexp). Requires A square and terms >= 1 (else
// domain_error). EXACT iff A is nilpotent and terms is at least its nilpotency index (t != 0
// preserves nilpotency; t == 0 gives T(0) = I exactly for any terms >= 1); otherwise an exact
// RATIONAL truncation of the transcendental e^{tA}. Overflow is propagated.
[[nodiscard]] auto semigroup(const Matrix& a, const Rational& t, std::int64_t terms)
    -> Result<Matrix>;

// The abstract Cauchy problem du/dt = A u, u(0) = u0: returns u(t) = T(t) u0. Requires A
// square (n x n), u0 an n x 1 column, and terms >= 1 (else domain_error). Exactness inherits
// semigroup()'s contract. Overflow is propagated.
[[nodiscard]] auto cauchy_solution(const Matrix& a, const Matrix& u0, const Rational& t,
                                   std::int64_t terms) -> Result<Matrix>;

// Verify the SEMIGROUP PROPERTY T(s+t) = T(s) T(t) at the given s, t (T(0) = I is the s=t=0
// case). Returns true iff the truncated operators satisfy it EXACTLY: for a nilpotent A with
// terms >= its nilpotency index this holds; for a truncated non-nilpotent A the identity is
// only approximate and this honestly returns false. Requires A square and terms >= 1 (else
// domain_error).
[[nodiscard]] auto verify_semigroup_property(const Matrix& a, const Rational& s,
                                             const Rational& t, std::int64_t terms)
    -> Result<bool>;

// Abstract-PDE connection. A linear evolution PDE u_t = L[u] becomes the abstract Cauchy
// problem u' = A u once L is replaced by its finite-dimensional spatial discretization A (the
// generator), e.g. from nimblecas.pde. Given that discretized generator and initial data u0,
// this returns the semigroup solution T(t) u0 -- identical to cauchy_solution, named for the
// PDE context. Requires A square (n x n), u0 an n x 1 column, terms >= 1 (else domain_error).
[[nodiscard]] auto pde_semigroup_solution(const Matrix& generator, const Matrix& u0,
                                          const Rational& t, std::int64_t terms)
    -> Result<Matrix>;

// ---------------------------------------------------------------------------
// Hille-Yosida / Lumer-Phillips (finite-dimensional realization).
// ---------------------------------------------------------------------------
// Every matrix A generates a C0-semigroup (the finite-dimensional Hille-Yosida statement),
// so the interesting content is the CONTRACTION / decay conditions, decided EXACTLY.

// Whether A is DISSIPATIVE: its symmetric part is negative semidefinite, A + A^T <= 0
// (equivalently x^T(A + A^T)x <= 0 for all x). Decided EXACTLY by testing that the symmetric
// matrix -(A + A^T) is POSITIVE semidefinite via a characteristic-polynomial coefficient-sign
// test: a real symmetric S with char. poly sum c_k lambda^k (monic, degree n) is positive
// semidefinite iff (-1)^{n+k} c_k >= 0 for all k. Like Routh-Hurwitz this settles irrational
// eigenvalues without root finding. Requires A square (else domain_error).
[[nodiscard]] auto is_dissipative(const Matrix& a) -> Result<bool>;

// Whether A generates a CONTRACTION semigroup (||T(t)|| <= 1 for all t >= 0). By Lumer-
// Phillips (finite-dimensional realization) this is exactly dissipativity, so this delegates
// to is_dissipative and returns an EXACT verdict. Requires A square (else domain_error).
[[nodiscard]] auto is_contraction_generator(const Matrix& a) -> Result<bool>;

// Whether A is HURWITZ: every eigenvalue has strictly negative real part, so ||T(t)|| -> 0
// as t -> infinity (uniform exponential stability). Decided EXACTLY by the Routh-Hurwitz
// criterion (via nimblecas.dynamics), so it settles irrational/complex spectra. Requires A
// square with n >= 1 (else domain_error).
[[nodiscard]] auto is_hurwitz(const Matrix& a) -> Result<bool>;

// ---------------------------------------------------------------------------
// Operator equations.
// ---------------------------------------------------------------------------

// Solve the SYLVESTER equation A X + X B = C EXACTLY over Q by vectorization: with column-
// stacking vec, (I_n (x) A + B^T (x) I_m) vec(X) = vec(C), an (m*n) x (m*n) exact rational
// linear system. Requires A square (m x m), B square (n x n), and C of shape m x n (else
// domain_error). The system is singular -- propagated as domain_error from the solve -- iff
// A and -B share an eigenvalue (no unique solution). Overflow is propagated.
[[nodiscard]] auto sylvester_solve(const Matrix& a, const Matrix& b, const Matrix& c)
    -> Result<Matrix>;

// The Lyapunov special case B = A^T: solve A X + X A^T = C EXACTLY over Q. Requires A square
// (n x n) and C of shape n x n (else domain_error). Singular iff A and -A^T share an
// eigenvalue (i.e. lambda_i + lambda_j = 0 for some eigenvalues), propagated as domain_error.
[[nodiscard]] auto lyapunov_equation(const Matrix& a, const Matrix& c) -> Result<Matrix>;

// VARIATION OF CONSTANTS for u' = A u + f(s), u(0) = u0, with a polynomial forcing
// f(s) = sum_j forcing[j] * s^j (each forcing[j] an n x 1 vector; an empty list means the
// homogeneous problem f = 0). Returns the closed form
//     u(t) = T(t) u0 + \int_0^t T(t-s) f(s) ds,
// with every operator series truncated at `terms` terms. EXACT iff A is nilpotent and terms
// is at least its nilpotency index (then every series is finite and the integral is closed-
// form over Q); otherwise the honest exact-rational truncation. Requires A square (n x n),
// u0 and each forcing[j] of shape n x 1, and terms >= 1 (else domain_error). Overflow is
// propagated. (Exponential forcing and other non-polynomial f are out of scope here and are
// left to the symbolic layer -- pass a polynomial or use the unevaluated form.)
[[nodiscard]] auto variation_of_constants(const Matrix& a, const Matrix& u0,
                                          const std::vector<Matrix>& forcing, const Rational& t,
                                          std::int64_t terms) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// |r| exactly. Denominators are kept positive, so the numerator carries the sign.
[[nodiscard]] auto rat_abs(const Rational& r) -> Result<Rational> {
    if (r.numerator() < 0) {
        return r.negate();
    }
    return r;
}

// a < b, exact (sign of a - b; canonical denominators are positive).
[[nodiscard]] auto rat_less(const Rational& a, const Rational& b) -> Result<bool> {
    auto d = a.subtract(b);
    if (!d) {
        return make_error<bool>(d.error());
    }
    return d->numerator() < 0;
}

// max(a, b), exact.
[[nodiscard]] auto rat_max(const Rational& a, const Rational& b) -> Result<Rational> {
    auto lt = rat_less(a, b);
    if (!lt) {
        return make_error<Rational>(lt.error());
    }
    return *lt ? b : a;
}

// Whether the SYMMETRIC rational matrix `m` is positive semidefinite (all eigenvalues >= 0),
// decided from the characteristic polynomial. For a real symmetric M all eigenvalues are
// real, and with p(lambda) = det(lambda*I - M) = sum_k c_k lambda^k (monic, degree n) one has
//   M PSD  <=>  every root >= 0  <=>  (-1)^{n+k} c_k >= 0 for all k
// (because (-1)^n p(-x) = prod (x + lambda_i) has all-nonnegative coefficients exactly when
// every lambda_i >= 0). This is an exact coefficient test -- no root finding -- so it handles
// irrational eigenvalues. PRECONDITION: `m` is symmetric; the caller guarantees this.
[[nodiscard]] auto is_psd_symmetric(const Matrix& m) -> Result<bool> {
    auto poly = characteristic_polynomial(m);
    if (!poly) {
        return make_error<bool>(poly.error());
    }
    const std::size_t n = m.rows();
    for (std::size_t k = 0; k <= n; ++k) {
        const Rational ck = poly->coefficient(k);
        const bool want_nonneg = ((n + k) % 2 == 0);  // sign required of c_k is (-1)^{n+k}
        if (want_nonneg) {
            if (ck.numerator() < 0) {
                return false;
            }
        } else {
            if (ck.numerator() > 0) {
                return false;
            }
        }
    }
    return true;
}

// Guard an r*c product against std::size_t wrap.
[[nodiscard]] auto fits(std::size_t r, std::size_t c) -> bool {
    return r == 0 || c <= std::numeric_limits<std::size_t>::max() / r;
}

}  // namespace

// --- operator / functional-analysis tools -----------------------------------

auto resolvent(const Matrix& a, const Rational& lambda) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    auto lambda_i = Matrix::identity(n).scale(lambda);  // lambda*I
    if (!lambda_i) {
        return make_error<Matrix>(lambda_i.error());
    }
    auto shifted = lambda_i->subtract(a);  // lambda*I - A
    if (!shifted) {
        return make_error<Matrix>(shifted.error());
    }
    return shifted->inverse();  // singular (lambda in spectrum) => domain_error
}

auto spectrum(const Matrix& a) -> Result<Spectrum> {
    if (!a.is_square()) {
        return make_error<Spectrum>(MathError::domain_error);
    }
    auto eig = rational_eigenvalues(a);
    if (!eig) {
        return make_error<Spectrum>(eig.error());
    }
    std::int64_t count = 0;
    for (const auto& [value, multiplicity] : *eig) {
        count += multiplicity;
    }
    const auto dim = static_cast<std::int64_t>(a.rows());
    return Spectrum{.rational_values = std::move(*eig),
                    .rational_count = count,
                    .dimension = dim,
                    .fully_extracted = (count == dim)};
}

auto spectral_radius(const Matrix& a) -> Result<SpectralRadius> {
    auto spec = spectrum(a);
    if (!spec) {
        return make_error<SpectralRadius>(spec.error());
    }
    if (spec->fully_extracted) {
        // Exact radius: the maximum |lambda| over the (complete) rational spectrum.
        Rational best = Rational::from_int(0);
        for (const auto& [value, multiplicity] : spec->rational_values) {
            auto abs_v = rat_abs(value);
            if (!abs_v) {
                return make_error<SpectralRadius>(abs_v.error());
            }
            auto m = rat_max(best, *abs_v);
            if (!m) {
                return make_error<SpectralRadius>(m.error());
            }
            best = *m;
        }
        return SpectralRadius{.value = best, .exact = true};
    }
    // Non-rational spectrum: fall back to a guaranteed rational upper bound rho(A) <= ||A||,
    // taking the tighter of the induced 1- and inf-norms.
    auto n1 = operator_norm_1(a);
    if (!n1) {
        return make_error<SpectralRadius>(n1.error());
    }
    auto ninf = operator_norm_inf(a);
    if (!ninf) {
        return make_error<SpectralRadius>(ninf.error());
    }
    auto lt = rat_less(*n1, *ninf);
    if (!lt) {
        return make_error<SpectralRadius>(lt.error());
    }
    return SpectralRadius{.value = (*lt ? *n1 : *ninf), .exact = false};
}

auto adjoint(const Matrix& a) -> Result<Matrix> {
    // Real rational operator: the adjoint is the transpose.
    return a.transpose();
}

auto operator_norm_1(const Matrix& a) -> Result<Rational> {
    Rational best = Rational::from_int(0);
    for (std::size_t j = 0; j < a.cols(); ++j) {
        Rational col_sum = Rational::from_int(0);
        for (std::size_t i = 0; i < a.rows(); ++i) {
            auto abs_ij = rat_abs(a.at(i, j));
            if (!abs_ij) {
                return make_error<Rational>(abs_ij.error());
            }
            auto s = col_sum.add(*abs_ij);
            if (!s) {
                return make_error<Rational>(s.error());
            }
            col_sum = *s;
        }
        auto m = rat_max(best, col_sum);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        best = *m;
    }
    return best;
}

auto operator_norm_inf(const Matrix& a) -> Result<Rational> {
    Rational best = Rational::from_int(0);
    for (std::size_t i = 0; i < a.rows(); ++i) {
        Rational row_sum = Rational::from_int(0);
        for (std::size_t j = 0; j < a.cols(); ++j) {
            auto abs_ij = rat_abs(a.at(i, j));
            if (!abs_ij) {
                return make_error<Rational>(abs_ij.error());
            }
            auto s = row_sum.add(*abs_ij);
            if (!s) {
                return make_error<Rational>(s.error());
            }
            row_sum = *s;
        }
        auto m = rat_max(best, row_sum);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        best = *m;
    }
    return best;
}

// --- C0-semigroup -----------------------------------------------------------

auto semigroup(const Matrix& a, const Rational& t, std::int64_t terms) -> Result<Matrix> {
    if (!a.is_square() || terms < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto ta = a.scale(t);  // t*A
    if (!ta) {
        return make_error<Matrix>(ta.error());
    }
    return matrix_exp_taylor(*ta, terms);  // e^{tA}, exact iff t*A nilpotent w/ enough terms
}

auto cauchy_solution(const Matrix& a, const Matrix& u0, const Rational& t, std::int64_t terms)
    -> Result<Matrix> {
    if (!a.is_square() || u0.rows() != a.rows() || u0.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto tt = semigroup(a, t, terms);
    if (!tt) {
        return make_error<Matrix>(tt.error());
    }
    return tt->multiply(u0);  // T(t) u0
}

auto verify_semigroup_property(const Matrix& a, const Rational& s, const Rational& t,
                               std::int64_t terms) -> Result<bool> {
    if (!a.is_square() || terms < 1) {
        return make_error<bool>(MathError::domain_error);
    }
    auto ts = semigroup(a, s, terms);
    if (!ts) {
        return make_error<bool>(ts.error());
    }
    auto tt = semigroup(a, t, terms);
    if (!tt) {
        return make_error<bool>(tt.error());
    }
    auto sum = s.add(t);
    if (!sum) {
        return make_error<bool>(sum.error());
    }
    auto tst = semigroup(a, *sum, terms);  // T(s+t)
    if (!tst) {
        return make_error<bool>(tst.error());
    }
    auto prod = ts->multiply(*tt);  // T(s) T(t)
    if (!prod) {
        return make_error<bool>(prod.error());
    }
    return tst->is_equal(*prod);
}

auto pde_semigroup_solution(const Matrix& generator, const Matrix& u0, const Rational& t,
                            std::int64_t terms) -> Result<Matrix> {
    // The discretized generator IS the finite-dimensional operator; the PDE solution is the
    // Cauchy solution of u' = A u, u(0) = u0.
    return cauchy_solution(generator, u0, t, terms);
}

// --- Hille-Yosida / Lumer-Phillips ------------------------------------------

auto is_dissipative(const Matrix& a) -> Result<bool> {
    if (!a.is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    auto at = a.transpose();
    if (!at) {
        return make_error<bool>(at.error());
    }
    auto sym = a.add(*at);  // symmetric part scaled by 2: M = A + A^T
    if (!sym) {
        return make_error<bool>(sym.error());
    }
    // A + A^T negative semidefinite  <=>  -(A + A^T) positive semidefinite.
    auto neg = sym->scale(Rational::from_int(-1));
    if (!neg) {
        return make_error<bool>(neg.error());
    }
    return is_psd_symmetric(*neg);
}

auto is_contraction_generator(const Matrix& a) -> Result<bool> {
    // Lumer-Phillips (finite-dim): contraction semigroup <=> dissipative generator.
    return is_dissipative(a);
}

auto is_hurwitz(const Matrix& a) -> Result<bool> {
    return is_asymptotically_stable(a);  // strict left half-plane spectrum (Routh-Hurwitz)
}

// --- operator equations -----------------------------------------------------

auto sylvester_solve(const Matrix& a, const Matrix& b, const Matrix& c) -> Result<Matrix> {
    if (!a.is_square() || !b.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t m = a.rows();
    const std::size_t n = b.rows();
    if (c.rows() != m || c.cols() != n) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (!fits(m, n)) {
        return make_error<Matrix>(MathError::overflow);
    }
    const std::size_t mn = m * n;

    // Build K = I_n (x) A + B^T (x) I_m, size (mn) x (mn), with column-stacking indexing
    // row = r*m + a, col = cc*m + bb (r,cc index the n blocks; a,bb index within m):
    //   K[r*m+a][cc*m+bb] = [r==cc] * A(a,bb) + [a==bb] * B(cc,r)   (B^T(r,cc) = B(cc,r)).
    std::vector<std::vector<Rational>> k_rows(mn, std::vector<Rational>(mn, Rational::from_int(0)));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t aa = 0; aa < m; ++aa) {
            const std::size_t row = r * m + aa;
            for (std::size_t cc = 0; cc < n; ++cc) {
                for (std::size_t bb = 0; bb < m; ++bb) {
                    const std::size_t col = cc * m + bb;
                    Rational val = Rational::from_int(0);
                    if (r == cc) {
                        auto s = val.add(a.at(aa, bb));
                        if (!s) {
                            return make_error<Matrix>(s.error());
                        }
                        val = *s;
                    }
                    if (aa == bb) {
                        auto s = val.add(b.at(cc, r));
                        if (!s) {
                            return make_error<Matrix>(s.error());
                        }
                        val = *s;
                    }
                    k_rows[row][col] = val;
                }
            }
        }
    }
    auto k = Matrix::from_rows(std::move(k_rows));
    if (!k) {
        return make_error<Matrix>(k.error());
    }

    // Right-hand side vec(C) (column-stacked): rhs[j*m + i] = C(i, j).
    std::vector<std::vector<Rational>> rhs_rows(mn, std::vector<Rational>(1, Rational::from_int(0)));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            rhs_rows[j * m + i][0] = c.at(i, j);
        }
    }
    auto rhs = Matrix::from_rows(std::move(rhs_rows));
    if (!rhs) {
        return make_error<Matrix>(rhs.error());
    }

    auto y = k->solve(*rhs);  // singular (A, -B share an eigenvalue) => domain_error
    if (!y) {
        return make_error<Matrix>(y.error());
    }

    // Un-stack: X(i, j) = y[j*m + i].
    std::vector<std::vector<Rational>> x_rows(m, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            x_rows[i][j] = y->at(j * m + i, 0);
        }
    }
    return Matrix::from_rows(std::move(x_rows));
}

auto lyapunov_equation(const Matrix& a, const Matrix& c) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto at = a.transpose();
    if (!at) {
        return make_error<Matrix>(at.error());
    }
    return sylvester_solve(a, *at, c);  // A X + X A^T = C
}

auto variation_of_constants(const Matrix& a, const Matrix& u0, const std::vector<Matrix>& forcing,
                            const Rational& t, std::int64_t terms) -> Result<Matrix> {
    if (!a.is_square() || terms < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (u0.rows() != n || u0.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    for (const auto& f : forcing) {
        if (f.rows() != n || f.cols() != 1) {
            return make_error<Matrix>(MathError::domain_error);
        }
    }

    // Homogeneous part: T(t) u0.
    auto homo = cauchy_solution(a, u0, t, terms);
    if (!homo) {
        return make_error<Matrix>(homo.error());
    }
    Matrix result = *homo;
    if (forcing.empty()) {
        return result;
    }

    // Particular part. With f(s) = sum_j c_j s^j and the substitution tau = t - s,
    //   \int_0^t T(t-s) f(s) ds = sum_j sum_{k>=0} A^k/k! * S(k,j) * c_j,
    //   S(k,j) = t^{k+j+1} * sum_{i=0}^{j} C(j,i) (-1)^i / (k+i+1),
    // truncating k at `terms` terms. For a nilpotent A with terms >= its index every A^k with
    // k >= index vanishes, so the finite sum is the exact closed form over Q.
    const std::int64_t deg = static_cast<std::int64_t>(forcing.size()) - 1;

    // Powers of t up to the largest exponent (terms-1) + deg + 1.
    const std::size_t max_exp =
        static_cast<std::size_t>(terms - 1) + static_cast<std::size_t>(deg) + 1;
    std::vector<Rational> tpow(max_exp + 1, Rational::from_int(1));  // tpow[0] = 1
    for (std::size_t p = 1; p <= max_exp; ++p) {
        auto next = tpow[p - 1].multiply(t);
        if (!next) {
            return make_error<Matrix>(next.error());
        }
        tpow[p] = *next;
    }

    Matrix power = Matrix::identity(n);           // A^k, k = 0, 1, ...
    Rational inv_fact = Rational::from_int(1);     // 1/k!
    for (std::int64_t k = 0; k < terms; ++k) {
        for (std::int64_t j = 0; j <= deg; ++j) {
            // inner = sum_{i=0}^{j} C(j,i)(-1)^i / (k+i+1), built with a running binomial.
            Rational inner = Rational::from_int(0);
            Rational binom = Rational::from_int(1);  // C(j,0)
            for (std::int64_t i = 0; i <= j; ++i) {
                auto denom = Rational::from_int(k + i + 1);
                auto term = binom.divide(denom);  // denom >= 1, never zero
                if (!term) {
                    return make_error<Matrix>(term.error());
                }
                if (i % 2 == 1) {
                    auto neg = term->negate();
                    if (!neg) {
                        return make_error<Matrix>(neg.error());
                    }
                    term = *neg;
                }
                auto s = inner.add(*term);
                if (!s) {
                    return make_error<Matrix>(s.error());
                }
                inner = *s;
                if (i < j) {
                    // C(j,i+1) = C(j,i) * (j - i) / (i + 1).
                    auto up = binom.multiply(Rational::from_int(j - i));
                    if (!up) {
                        return make_error<Matrix>(up.error());
                    }
                    auto down = up->divide(Rational::from_int(i + 1));
                    if (!down) {
                        return make_error<Matrix>(down.error());
                    }
                    binom = *down;
                }
            }

            // scalar = (1/k!) * inner * t^{k+j+1}.
            auto sc1 = inv_fact.multiply(inner);
            if (!sc1) {
                return make_error<Matrix>(sc1.error());
            }
            const std::size_t exp = static_cast<std::size_t>(k + j + 1);
            auto scalar = sc1->multiply(tpow[exp]);
            if (!scalar) {
                return make_error<Matrix>(scalar.error());
            }
            if (!scalar->is_zero()) {
                auto pcj = power.multiply(forcing[static_cast<std::size_t>(j)]);  // A^k c_j
                if (!pcj) {
                    return make_error<Matrix>(pcj.error());
                }
                auto contrib = pcj->scale(*scalar);
                if (!contrib) {
                    return make_error<Matrix>(contrib.error());
                }
                auto sum = result.add(*contrib);
                if (!sum) {
                    return make_error<Matrix>(sum.error());
                }
                result = *sum;
            }
        }

        // Advance A^k -> A^{k+1} and 1/k! -> 1/(k+1)!, only if another term is needed.
        if (k + 1 < terms) {
            auto next_power = power.multiply(a);
            if (!next_power) {
                return make_error<Matrix>(next_power.error());
            }
            power = *next_power;
            auto next_inv = inv_fact.divide(Rational::from_int(k + 1));
            if (!next_inv) {
                return make_error<Matrix>(next_inv.error());
            }
            inv_fact = *next_inv;
        }
    }
    return result;
}

}  // namespace nimblecas
