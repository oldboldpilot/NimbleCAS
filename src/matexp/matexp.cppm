// NimbleCAS exact-rational matrix exponentials (ROADMAP 7.2/7.19).
// @author Olumuyiwa Oluwasanmi
//
// Exact-over-the-rationals routines for the matrix exponential e^A, the object that closes
// the linear-ODE story: the solution of dx/dt = A x is x(t) = e^{At} x0. Everything here
// stays inside Rational arithmetic and rides the Result railway; dimension violations and
// bad parameters surface as MathError::domain_error, and any Rational overflow is propagated.
//
// HONESTY ABOUT EXACTNESS. The true matrix exponential e^A is generally TRANSCENDENTAL and
// therefore not representable over Q. What these routines return exactly is:
//
//   matrix_exp_taylor(A, terms) -- the truncated Taylor series  Sum_{k=0}^{terms-1} A^k / k!.
//       This equals e^A EXACTLY iff A is nilpotent AND terms exceeds the nilpotency index
//       (then the series terminates: every omitted term is a power of A that is already zero).
//       For a non-nilpotent A it is only a rational polynomial approximation.
//   matrix_exp_pade(A, q)       -- the diagonal [q/q] Pade approximant  D^{-1} N, a rational
//       (matrix-valued) approximation of e^A. It too is EXACT when A is nilpotent (numerator
//       and denominator polynomials both terminate at the nilpotency index and reproduce the
//       finite series). Otherwise it is a distinct rational approximation from Taylor's.
//   matrix_exp(A, q, s)         -- scaling-and-squaring: e^A = (e^{A/2^s})^{2^s}, with the
//       inner factor formed by the [q/q] Pade approximant of A/2^s. This is the numerically
//       standard route for e^A; it is likewise EXACT for nilpotent A (scaling preserves
//       nilpotency and the repeated squaring reproduces the exact answer).
//
// So: nilpotent inputs give genuinely exact matrix exponentials; every other input yields an
// exact RATIONAL APPROXIMATION (Taylor or Pade), never a claim of the transcendental truth.

export module nimblecas.matexp;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// Whether A is nilpotent: true iff A^n is the zero matrix, where n = A.rows(). For an n x n
// matrix this n-th power is a complete test (a nilpotent matrix has nilpotency index at most
// n, so if any power vanishes then A^n already does). Requires A square, else domain_error.
// A^n is formed by repeated exact multiplication, so overflow in an entry is propagated.
[[nodiscard]] auto is_nilpotent(const Matrix& a) -> Result<bool>;

// The truncated Taylor series  Sum_{k=0}^{terms-1} A^k / k!  of e^A, exact over Rational.
// Requires A square and terms >= 1, else domain_error. A running power P (P_0 = I,
// P_k = P_{k-1} * A) and a running 1/k! are accumulated so no large factorial is formed.
// EXACT e^A precisely when A is nilpotent and terms exceeds the nilpotency index (the tail
// is then all zero); otherwise a rational polynomial approximation. Overflow is propagated.
[[nodiscard]] auto matrix_exp_taylor(const Matrix& a, std::int64_t terms) -> Result<Matrix>;

// The diagonal [q/q] Pade approximant of e^A: with scalar coefficients c_k (c_0 = 1),
// N = Sum_{k=0}^{q} c_k A^k  and  D = Sum_{k=0}^{q} (-1)^k c_k A^k, returning D^{-1} N.
// The c_k are built by the ratio recurrence c_k = c_{k-1} * (q-k+1) / (k*(2q-k+1)) to avoid
// forming huge factorials. Requires A square and q >= 1, else domain_error. A singular D
// propagates domain_error from the inverse. EXACT for nilpotent A; otherwise a rational
// approximation of e^A. Rational overflow is propagated.
[[nodiscard]] auto matrix_exp_pade(const Matrix& a, std::size_t q) -> Result<Matrix>;

// Scaling-and-squaring exponential (the recommended route): with s = scaling_power, form
// B = A / 2^s exactly, take X = matrix_exp_pade(B, q), then square X back s times
// (X <- X*X, s times) to recover an approximation of e^A. Requires A square and q >= 1,
// else domain_error; s may be 0 (then it coincides with matrix_exp_pade(A, q)). This is the
// numerically standard method and is still EXACT for nilpotent A. Overflow (including in the
// 2^s scale factor) is propagated.
[[nodiscard]] auto matrix_exp(const Matrix& a, std::size_t q, std::size_t scaling_power)
    -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// acc + power * coeff, exact. Used to fold one series term into a running sum.
[[nodiscard]] auto add_scaled(const Matrix& acc, const Matrix& power, const Rational& coeff)
    -> Result<Matrix> {
    auto term = power.scale(coeff);
    if (!term) {
        return make_error<Matrix>(term.error());
    }
    return acc.add(*term);
}

}  // namespace

auto is_nilpotent(const Matrix& a) -> Result<bool> {
    if (!a.is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // Compute A^n by repeated multiplication (A^0 = I) and test against the zero matrix.
    Matrix power = Matrix::identity(n);
    for (std::size_t k = 0; k < n; ++k) {
        auto next = power.multiply(a);
        if (!next) {
            return make_error<bool>(next.error());
        }
        power = *next;
    }
    return power.is_equal(Matrix::zero(n, n));
}

auto matrix_exp_taylor(const Matrix& a, std::int64_t terms) -> Result<Matrix> {
    if (!a.is_square() || terms < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // Accumulate Sum_{k=0}^{terms-1} A^k / k! with a running power P_k = A^k and a running
    // reciprocal factorial inv_fact = 1/k! (inv_fact_k = inv_fact_{k-1} / k).
    Matrix power = Matrix::identity(n);
    Matrix sum = Matrix::zero(n, n);
    Rational inv_fact = Rational::from_int(1);  // 1/0! = 1
    for (std::int64_t k = 0; k < terms; ++k) {
        auto folded = add_scaled(sum, power, inv_fact);
        if (!folded) {
            return make_error<Matrix>(folded.error());
        }
        sum = *folded;

        // Advance to the next term only if one is still needed (avoids a spurious final
        // multiply / factorial step that could overflow without being used).
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
    return sum;
}

auto matrix_exp_pade(const Matrix& a, std::size_t q) -> Result<Matrix> {
    if (!a.is_square() || q < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // N = Sum c_k A^k, D = Sum (-1)^k c_k A^k. Build c_k by the ratio recurrence
    //   c_0 = 1,  c_k = c_{k-1} * (q-k+1) / (k * (2q-k+1)),
    // stepped through exact Rational multiply/divide (each factor kept small to dodge the
    // factorial overflow the closed form would incur).
    Matrix power = Matrix::identity(n);
    Matrix num = Matrix::zero(n, n);
    Matrix den = Matrix::zero(n, n);
    Rational c = Rational::from_int(1);  // c_0
    for (std::size_t k = 0; k <= q; ++k) {
        // Numerator uses c_k; denominator uses (-1)^k c_k.
        Rational c_signed = c;
        if (k % 2 == 1) {
            auto neg = c.negate();
            if (!neg) {
                return make_error<Matrix>(neg.error());
            }
            c_signed = *neg;
        }
        auto folded_num = add_scaled(num, power, c);
        if (!folded_num) {
            return make_error<Matrix>(folded_num.error());
        }
        num = *folded_num;
        auto folded_den = add_scaled(den, power, c_signed);
        if (!folded_den) {
            return make_error<Matrix>(folded_den.error());
        }
        den = *folded_den;

        if (k < q) {
            auto next_power = power.multiply(a);
            if (!next_power) {
                return make_error<Matrix>(next_power.error());
            }
            power = *next_power;

            // Step to c_{k+1}: multiply by (q - (k+1) + 1) = (q - k), then divide by the two
            // small factors (k+1) and (2q - k). Doing it in three steps keeps every operand
            // small so no intermediate integer product overflows.
            auto s1 = c.multiply(Rational::from_int(static_cast<std::int64_t>(q - k)));
            if (!s1) {
                return make_error<Matrix>(s1.error());
            }
            auto s2 = s1->divide(Rational::from_int(static_cast<std::int64_t>(k + 1)));
            if (!s2) {
                return make_error<Matrix>(s2.error());
            }
            auto s3 = s2->divide(Rational::from_int(static_cast<std::int64_t>(2 * q - k)));
            if (!s3) {
                return make_error<Matrix>(s3.error());
            }
            c = *s3;
        }
    }

    // e^A ~ D^{-1} N. A singular D (never happens for the standard exp coefficients, but
    // guarded regardless) surfaces as domain_error from inverse().
    auto den_inv = den.inverse();
    if (!den_inv) {
        return make_error<Matrix>(den_inv.error());
    }
    return den_inv->multiply(num);
}

auto matrix_exp(const Matrix& a, std::size_t q, std::size_t scaling_power) -> Result<Matrix> {
    if (!a.is_square() || q < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }

    // Scale factor 1 / 2^s, with 2^s formed by repeated doubling under an overflow guard.
    std::int64_t pow2 = 1;
    for (std::size_t i = 0; i < scaling_power; ++i) {
        if (pow2 > std::numeric_limits<std::int64_t>::max() / 2) {
            return make_error<Matrix>(MathError::overflow);
        }
        pow2 *= 2;
    }
    auto scale_factor = Rational::make(1, pow2);  // 1 / 2^s
    if (!scale_factor) {
        return make_error<Matrix>(scale_factor.error());
    }
    auto scaled = a.scale(*scale_factor);
    if (!scaled) {
        return make_error<Matrix>(scaled.error());
    }

    // Inner Pade approximant of B = A / 2^s, then square back s times: (e^B)^{2^s} = e^A.
    auto x = matrix_exp_pade(*scaled, q);
    if (!x) {
        return make_error<Matrix>(x.error());
    }
    Matrix result = *x;
    for (std::size_t i = 0; i < scaling_power; ++i) {
        auto squared = result.multiply(result);
        if (!squared) {
            return make_error<Matrix>(squared.error());
        }
        result = *squared;
    }
    return result;
}

}  // namespace nimblecas
