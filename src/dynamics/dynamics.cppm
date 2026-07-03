// NimbleCAS dynamical-systems stability (ROADMAP 7.10).
// @author Olumuyiwa Oluwasanmi
//
// Exact stability analysis of linear/affine autonomous systems dx/dt = A x (+ b),
// built entirely on top of the rational eigen substrate. Three entry points:
//
//   fixed_point_affine(A, b)     -- the equilibrium of x |-> A x + b, i.e. the exact
//                                   solution of (I - A) x = b over Q.
//   is_asymptotically_stable(A)  -- true iff every eigenvalue of A has strictly negative
//                                   real part, decided EXACTLY from the characteristic
//                                   polynomial via the Routh-Hurwitz criterion. Because it
//                                   works on coefficients (no root finding), it settles the
//                                   irrational/complex spectra that rational-root testing
//                                   over Q cannot -- e.g. the pure imaginary pair of a
//                                   rotation. That is precisely why Routh-Hurwitz is used.
//   classify_equilibrium(A)      -- a stable, human-readable classification string.
//
// Everything stays inside exact Rational arithmetic; every fallible step is threaded on
// the Result railway and dimension violations surface as MathError::domain_error.

export module nimblecas.dynamics;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;

export namespace nimblecas {

// The equilibrium (fixed point) of the affine map x |-> A x + b, i.e. the exact solution
// of (I - A) x = b. `A` must be square (n x n) and `b` an n x 1 column matrix, otherwise
// domain_error. A non-isolated equilibrium -- where (I - A) is singular -- propagates as
// domain_error from the underlying solve. On success the returned n x 1 column is x.
[[nodiscard]] auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// Whether the continuous-time system dx/dt = A x is asymptotically stable: true iff every
// eigenvalue of A has strictly negative real part (Hurwitz). Decided EXACTLY from the
// characteristic polynomial of A via the Routh-Hurwitz criterion -- the Routh array is
// built with exact Rational arithmetic and the system is Hurwitz-stable iff every entry
// of the array's first column is nonzero and shares the leading term's (positive) sign.
// No root finding is involved, so this decides irrational/complex spectra too.
//
// Edge cases, both of which mean at least one root lies on or right of the imaginary axis
// and so are reported as NOT asymptotically stable (false): a zero in the first column
// (whether or not the rest of that row is nonzero), and a fully-zero row (imaginary-axis
// roots). Requires a square matrix with n >= 1, else domain_error.
[[nodiscard]] auto is_asymptotically_stable(const Matrix& a) -> Result<bool>;

// A human-readable classification of the equilibrium at the origin of dx/dt = A x.
//
// When ALL eigenvalues are rational (their multiplicities from rational_eigenvalues sum to
// n) the sign pattern gives an exact verdict:
//   - every eigenvalue negative        -> "stable node"
//   - every eigenvalue positive        -> "unstable node"
//   - both signs present (no zero)     -> "saddle"
//   - any zero eigenvalue present      -> "degenerate/marginal"
// (A zero eigenvalue is checked first: its marginal direction dominates the verdict.)
//
// Otherwise part of the spectrum is irrational/complex and cannot be seen over Q, so the
// verdict falls back to the Routh-Hurwitz result:
//   - asymptotically stable            -> "asymptotically stable (spiral/node)"
//   - not asymptotically stable        -> "unstable or marginal (non-rational spectrum)"
// (Rational eigenvalues cannot detect complex conjugate pairs, so "center"/"focus" is
// never claimed from them.) Requires a square matrix, else domain_error.
[[nodiscard]] auto classify_equilibrium(const Matrix& a) -> Result<std::string>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// -1, 0, +1 for a Rational's sign (denominator is kept positive in canonical form, so the
// numerator's sign is the value's sign).
[[nodiscard]] auto sign_of(const Rational& r) -> int {
    if (r.numerator() > 0) {
        return 1;
    }
    if (r.numerator() < 0) {
        return -1;
    }
    return 0;
}

// One Routh-array entry: (pivot * up_next - up0 * cur_next) / pivot, all exact. `pivot` is
// the first-column entry of the row above (known nonzero at the call site). Any Rational
// overflow/division error is propagated.
[[nodiscard]] auto routh_entry(const Rational& pivot, const Rational& up_next,
                               const Rational& up0, const Rational& cur_next)
    -> Result<Rational> {
    auto lhs = pivot.multiply(up_next);
    if (!lhs) {
        return make_error<Rational>(lhs.error());
    }
    auto rhs = up0.multiply(cur_next);
    if (!rhs) {
        return make_error<Rational>(rhs.error());
    }
    auto num = lhs->subtract(*rhs);
    if (!num) {
        return make_error<Rational>(num.error());
    }
    return num->divide(pivot);
}

}  // namespace

auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (b.rows() != a.rows() || b.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);
    }

    // Solve (I - A) x = b. A singular (I - A) -- a non-isolated equilibrium -- surfaces as
    // domain_error from solve, which we simply propagate.
    const Matrix id = Matrix::identity(a.rows());
    auto shifted = id.subtract(a);
    if (!shifted) {
        return make_error<Matrix>(shifted.error());
    }
    return shifted->solve(b);
}

auto is_asymptotically_stable(const Matrix& a) -> Result<bool> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<bool>(MathError::domain_error);
    }

    auto poly = characteristic_polynomial(a);
    if (!poly) {
        return make_error<bool>(poly.error());
    }
    const std::size_t n = a.rows();  // characteristic polynomial is monic of degree n

    // Coefficients in DESCENDING order: a_desc[i] is the coefficient of lambda^(n-i), so
    // a_desc[0] is the (monic, = 1) leading coefficient and a_desc[n] the constant term.
    std::vector<Rational> a_desc(n + 1, Rational::from_int(0));
    for (std::size_t i = 0; i <= n; ++i) {
        a_desc[i] = poly->coefficient(n - i);
    }

    // Routh array: n + 1 rows, each `cols` wide (padded with zeros). Row 0 holds the even-
    // indexed coefficients, row 1 the odd-indexed ones.
    const std::size_t cols = n / 2 + 1;
    std::vector<std::vector<Rational>> table(n + 1,
                                             std::vector<Rational>(cols, Rational::from_int(0)));
    for (std::size_t j = 0; j < cols; ++j) {
        if (const std::size_t e = 2 * j; e <= n) {
            table[0][j] = a_desc[e];
        }
        if (const std::size_t o = 2 * j + 1; o <= n) {
            table[1][j] = a_desc[o];
        }
    }

    // Fill rows 2..n. A zero pivot (first-column entry of the row above) means either a
    // zero in the first column or a fully-zero row -- both put a root on or right of the
    // imaginary axis, so the system is not asymptotically stable.
    for (std::size_t r = 2; r <= n; ++r) {
        const Rational pivot = table[r - 1][0];
        if (pivot.is_zero()) {
            return false;
        }
        for (std::size_t j = 0; j < cols; ++j) {
            const Rational up_next = (j + 1 < cols) ? table[r - 2][j + 1] : Rational::from_int(0);
            const Rational cur_next = (j + 1 < cols) ? table[r - 1][j + 1] : Rational::from_int(0);
            auto entry = routh_entry(pivot, up_next, table[r - 2][0], cur_next);
            if (!entry) {
                return make_error<bool>(entry.error());
            }
            table[r][j] = *entry;
        }
    }

    // Hurwitz iff every first-column entry is nonzero and shares the leading (positive)
    // sign. A zero here (e.g. a fully-zero final row that was never a pivot) is a failure.
    const int reference = sign_of(table[0][0]);  // +1, since the polynomial is monic
    for (std::size_t r = 0; r <= n; ++r) {
        const int s = sign_of(table[r][0]);
        if (s == 0 || s != reference) {
            return false;
        }
    }
    return true;
}

auto classify_equilibrium(const Matrix& a) -> Result<std::string> {
    if (!a.is_square()) {
        return make_error<std::string>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    auto eigenvalues = rational_eigenvalues(a);
    if (!eigenvalues) {
        return make_error<std::string>(eigenvalues.error());
    }

    // Do the rational eigenvalues account for the whole spectrum (multiplicities sum to n)?
    std::int64_t total_multiplicity = 0;
    bool has_negative = false;
    bool has_positive = false;
    bool has_zero = false;
    for (const auto& [value, multiplicity] : *eigenvalues) {
        total_multiplicity += multiplicity;
        switch (sign_of(value)) {
            case -1: has_negative = true; break;
            case 1:  has_positive = true; break;
            default: has_zero = true; break;
        }
    }

    if (total_multiplicity == static_cast<std::int64_t>(n)) {
        // Fully rational spectrum: classify by sign pattern.
        if (has_zero) {
            return std::string("degenerate/marginal");
        }
        if (has_negative && has_positive) {
            return std::string("saddle");
        }
        if (has_negative) {
            return std::string("stable node");
        }
        if (has_positive) {
            return std::string("unstable node");
        }
        // Empty spectrum (n == 0): no motion, treat as marginal.
        return std::string("degenerate/marginal");
    }

    // Irrational/complex part of the spectrum: defer to the exact Routh-Hurwitz verdict.
    auto stable = is_asymptotically_stable(a);
    if (!stable) {
        return make_error<std::string>(stable.error());
    }
    return *stable ? std::string("asymptotically stable (spiral/node)")
                   : std::string("unstable or marginal (non-rational spectrum)");
}

}  // namespace nimblecas
