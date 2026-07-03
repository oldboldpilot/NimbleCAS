// NimbleCAS Pade approximants (ROADMAP 7.4/7.5 resummation).
// @author Olumuyiwa Oluwasanmi
//
// The [m/n] Pade approximant of a formal power series s = c_0 + c_1 x + ... is the
// rational function P(x)/Q(x) with deg P <= m, deg Q <= n, normalised so Q(0) = 1,
// whose Maclaurin expansion agrees with s through the term x^{m+n}. It is the
// workhorse of asymptotic resummation: where a truncated Taylor polynomial diverges
// past its radius of convergence, the rational Pade form frequently continues to
// track the underlying function (and can reproduce it exactly when the function is
// itself rational, recovering P and Q up to the Q(0)=1 normalisation).
//
// Everything is computed exactly over Q. Q's unknown coefficients q_1..q_n (with
// q_0 = 1 fixed) come from the n linear equations that force the coefficients of
// x^{m+1}..x^{m+n} in the product Q*s to vanish; this is a Toeplitz n x n system
// A q = b assembled from the series coefficients and solved by the exact rational
// matrix solver. P is then read off directly from the low-order coefficients of Q*s.
// A degenerate (singular) system means the requested approximant does not exist and
// surfaces as MathError::domain_error, propagated from the solver (Rule 32 railway).

export module nimblecas.pade;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;

export namespace nimblecas {

// Compute the [m/n] Pade approximant of the truncated power series `s`, returning the
// pair (P, Q) of numerator and denominator as RationalPoly in ascending-degree
// coefficient order, with Q normalised to constant term 1 (Q(0) = 1).
//
// Requirements and failure modes (all MathError::domain_error):
//   * `s.order() >= m + n + 1` — the coefficients c_0..c_{m+n} must all be present;
//     a shorter series cannot pin down an [m/n] approximant.
//   * The n x n Toeplitz system for q_1..q_n must be nonsingular; a singular system
//     (the approximant of this order is degenerate / does not exist) propagates the
//     domain_error raised by Matrix::solve.
//
// The n = 0 case is handled directly: Q = 1 and P is simply the degree-m truncation
// of the series (p_k = c_k for k = 0..m), with no linear system to solve.
[[nodiscard]] auto pade(const PowerSeries& s, std::size_t m, std::size_t n)
    -> Result<std::pair<RationalPoly, RationalPoly>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto pade(const PowerSeries& s, std::size_t m, std::size_t n)
    -> Result<std::pair<RationalPoly, RationalPoly>> {
    using Pair = std::pair<RationalPoly, RationalPoly>;

    // Need c_0..c_{m+n}, i.e. order() >= m + n + 1. A PowerSeries always has order >= 1,
    // so max_index below never underflows; the bound is written to avoid any size_t
    // wrap in the sum m + n + 1.
    const std::size_t max_index = s.order() - 1;
    if (m > max_index || n > max_index - m) {
        return make_error<Pair>(MathError::domain_error);
    }

    const Rational one = Rational::from_int(1);

    // Coefficient c_i of the series, with the convention c_i = 0 for i < 0 (signed
    // index so the Toeplitz assembly below can reach below the constant term).
    auto coeff = [&](std::int64_t i) -> Rational {
        if (i < 0) {
            return Rational{};  // 0
        }
        return s.coefficient(static_cast<std::size_t>(i));
    };

    // Denominator coefficients q_0..q_n, with q_0 = 1 fixed.
    std::vector<Rational> q(n + 1);
    q[0] = one;

    if (n >= 1) {
        // Assemble the n x n system A q' = b (q' = (q_1..q_n)) enforcing, for
        // k = m+1..m+n:  sum_{j=0..n} q_j c_{k-j} = 0. With q_0 = 1 this gives, for
        // rows r = 0..n-1 (equation k = m+1+r):
        //     A[r][cc] = c_{m + r - cc}      (cc = 0..n-1, i.e. the q_{cc+1} column)
        //     b[r]     = -c_{m + 1 + r}.
        std::vector<std::vector<Rational>> arows(n, std::vector<Rational>(n));
        std::vector<std::vector<Rational>> brows(n, std::vector<Rational>(1));
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t cc = 0; cc < n; ++cc) {
                arows[r][cc] = coeff(static_cast<std::int64_t>(m) +
                                     static_cast<std::int64_t>(r) -
                                     static_cast<std::int64_t>(cc));
            }
            auto neg = coeff(static_cast<std::int64_t>(m) + static_cast<std::int64_t>(r) + 1)
                           .negate();
            if (!neg) {
                return make_error<Pair>(neg.error());
            }
            brows[r][0] = *neg;
        }
        auto a = Matrix::from_rows(std::move(arows));
        if (!a) {
            return make_error<Pair>(a.error());
        }
        auto b = Matrix::from_rows(std::move(brows));
        if (!b) {
            return make_error<Pair>(b.error());
        }
        auto x = a->solve(*b);  // singular Toeplitz system => domain_error (propagated)
        if (!x) {
            return make_error<Pair>(x.error());
        }
        for (std::size_t r = 0; r < n; ++r) {
            q[r + 1] = x->at(r, 0);
        }
    }

    // Numerator coefficients: p_k = sum_{j=0..min(k,n)} q_j c_{k-j} for k = 0..m,
    // i.e. the low-order (degree <= m) coefficients of the product Q * s.
    std::vector<Rational> p(m + 1);
    for (std::size_t k = 0; k <= m; ++k) {
        Rational acc;  // 0
        const std::size_t jmax = std::min(k, n);
        for (std::size_t j = 0; j <= jmax; ++j) {
            auto term = q[j].multiply(s.coefficient(k - j));
            if (!term) {
                return make_error<Pair>(term.error());
            }
            auto sum = acc.add(*term);
            if (!sum) {
                return make_error<Pair>(sum.error());
            }
            acc = *sum;
        }
        p[k] = acc;
    }

    return Pair{RationalPoly::from_coeffs(std::move(p)),
                RationalPoly::from_coeffs(std::move(q))};
}

}  // namespace nimblecas
