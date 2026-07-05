// NimbleCAS linear homogeneous constant-coefficient recurrences (ROADMAP 7.9).
// @author Olumuyiwa Oluwasanmi
//
// A first slice of difference equations / recurrence relations. Given a linear
// homogeneous recurrence with constant rational coefficients
//
//     a_n = c_0 a_{n-1} + c_1 a_{n-2} + ... + c_{k-1} a_{n-k}   (k = order),
//
// its behaviour is governed by the characteristic polynomial
//
//     x^k - c_0 x^{k-1} - c_1 x^{k-2} - ... - c_{k-1}
//
// (monic, degree k). Each distinct root r of this polynomial, with multiplicity m,
// contributes a family of basis solutions r^n, n r^n, ..., n^{m-1} r^n to the general
// closed form a_n = sum over roots of (polynomial of degree m-1 in n) * r^n; the
// coefficients of those polynomials are fixed by the initial conditions.
//
// SCOPE: only the RATIONAL-characteristic-root case is fully resolved here. The rational
// roots (with multiplicities) are found exactly via nimblecas.roots::rational_roots on
// the characteristic polynomial, and all_roots_rational() reports whether the polynomial
// splits completely over Q (so the closed form is expressible with rational bases alone).
// When it does not — the Fibonacci recurrence x^2 - x - 1 is the canonical example, whose
// roots are the irrational golden-ratio conjugates — the closed form requires irrational
// or complex roots. Radical / RootOf closed forms for that case are a planned extension
// (mirroring the same documented limitation in nimblecas.roots) and are not produced here;
// characteristic_roots() then returns only the rational roots (possibly none).
//
// Two consumers of that spectral data are provided on top:
//
//   * closed_form() assembles the explicit solution a_n = sum over roots of
//     (c_{i,0} + c_{i,1} n + ... + c_{i,m_i-1} n^{m_i-1}) r_i^n as a symbolic Expr in n,
//     pinning the constants c_{i,j} by solving the (confluent-Vandermonde) linear system
//     from the initial conditions EXACTLY over Q. This needs the roots, so it is only
//     available in the rational-root case: when the characteristic polynomial does not
//     split over Q (the Fibonacci recurrence x^2 - x - 1 is the canonical example) it
//     returns MathError::not_implemented rather than emitting a wrong closed form.
//
//   * generating_function() returns the ordinary generating function G(x) = P(x)/Q(x)
//     with Q(x) = 1 - c_0 x - c_1 x^2 - ... - c_{k-1} x^k (the reflected characteristic
//     polynomial) and P(x) fixed by the initial conditions. This is exact and does NOT
//     need the roots, so — unlike the closed form — it is produced even when the roots
//     are irrational (its advantage): the Fibonacci GF is x / (1 - x - x^2).
//
// Following the rest of the engine, arithmetic is exact and overflow-checked and every
// fallible step is threaded through Result (Rule 32).

module;
#include <cassert>

export module nimblecas.recurrence;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.roots;
import nimblecas.symbolic;
import nimblecas.matrix;

export namespace nimblecas {

// The characteristic polynomial x^k - c_0 x^{k-1} - c_1 x^{k-2} - ... - c_{k-1} of the
// order-k recurrence a_n = c_0 a_{n-1} + ... + c_{k-1} a_{n-k}, returned as a monic
// RationalPoly of degree k. The coefficients are given low-order first, so coeffs[0] is
// c_0 (the a_{n-1} weight) and coeffs.back() is c_{k-1} (the a_{n-k} weight). Empty
// coeffs describe no recurrence and are rejected with MathError::domain_error. Fails only
// on overflow of the exact rational negation.
[[nodiscard]] auto characteristic_polynomial(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;

// The distinct RATIONAL characteristic roots of the recurrence, each paired with its
// multiplicity (>= 1), obtained by running rational_roots() on the characteristic
// polynomial. A distinct rational root r of multiplicity m contributes the basis terms
// r^n, n r^n, ..., n^{m-1} r^n to the general solution. Roots are returned in no
// particular order. Irrational / complex roots are NOT returned (see the module header on
// scope): for a recurrence whose characteristic polynomial does not split over Q this
// yields only the rational part, possibly the empty vector. Empty coeffs ->
// MathError::domain_error.
[[nodiscard]] auto characteristic_roots(std::span<const Rational> coeffs)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

// Whether the characteristic polynomial splits completely over Q — i.e. whether the sum
// of the rational-root multiplicities equals the order k. When true, the closed form is
// fully expressible with rational bases and characteristic_roots() accounts for every
// root. When false, the remaining roots are irrational or complex and the closed form
// needs radical / RootOf machinery that is a planned extension. Empty coeffs ->
// MathError::domain_error.
[[nodiscard]] auto all_roots_rational(std::span<const Rational> coeffs) -> Result<bool>;

// The explicit closed-form solution a(n) of the order-k recurrence
// a_n = c_0 a_{n-1} + ... + c_{k-1} a_{n-k} with the given initial conditions
// a_0, ..., a_{k-1} (initial[i] = a_i), returned as a symbolic Expr in the symbol "n".
//
// When the characteristic polynomial splits over Q (all_roots_rational), each distinct
// rational root r_i of multiplicity m_i contributes a term
// (c_{i,0} + c_{i,1} n + ... + c_{i,m_i-1} n^{m_i-1}) r_i^n; the k constants c_{i,j} are
// determined EXACTLY by solving the confluent-Vandermonde linear system whose row n
// (n = 0..k-1) enforces a(n) = a_n. The returned Expr is a(n) = sum over roots of those
// terms.
//
// `coeffs` is the recurrence coefficient list {c_0, ..., c_{k-1}} (low-order first, as in
// characteristic_polynomial) and `initial` must hold EXACTLY k = coeffs.size() initial
// conditions {a_0, ..., a_{k-1}}. Errors:
//   * empty `coeffs`, or initial.size() != coeffs.size()  -> MathError::domain_error;
//   * characteristic polynomial does NOT split over Q (irrational / complex roots, e.g.
//     Fibonacci x^2 - x - 1)                               -> MathError::not_implemented
//     (honest: no wrong closed form is emitted — use generating_function instead);
//   * overflow of the exact rational arithmetic, or a singular solve  -> propagated.
[[nodiscard]] auto closed_form(std::span<const Rational> coeffs,
                               std::span<const Rational> initial) -> Result<Expr>;

// The ordinary generating function G(x) = sum_{n>=0} a_n x^n of the recurrence, returned
// in exact rational form as G(x) = numerator / denominator with
//   denominator = Q(x) = 1 - c_0 x - c_1 x^2 - ... - c_{k-1} x^k   (degree k)
//   numerator   = P(x) = sum_{m=0}^{k-1} ( a_m - sum_{j=1}^{m} c_{j-1} a_{m-j} ) x^m.
// Q is the reflected characteristic polynomial; P is fixed by the initial conditions so
// that G*Q agrees with the sequence. This is EXACT and needs no roots, so it is produced
// even when the characteristic polynomial does not split over Q — its advantage over the
// closed form (the Fibonacci GF is x / (1 - x - x^2)).
struct GeneratingFunction {
    RationalPoly numerator;    // P(x)
    RationalPoly denominator;  // Q(x) = 1 - c_0 x - ... - c_{k-1} x^k
};

// See GeneratingFunction. `coeffs` = {c_0, ..., c_{k-1}} (low-order first); `initial` must
// hold EXACTLY k = coeffs.size() initial conditions. Empty `coeffs` or a mismatched
// `initial` length -> MathError::domain_error; overflow of the exact rational negation /
// convolution is propagated.
[[nodiscard]] auto generating_function(std::span<const Rational> coeffs,
                                       std::span<const Rational> initial)
    -> Result<GeneratingFunction>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto characteristic_polynomial(std::span<const Rational> coeffs) -> Result<RationalPoly> {
    if (coeffs.empty()) {
        return make_error<RationalPoly>(MathError::domain_error);  // no recurrence
    }
    const std::size_t k = coeffs.size();
    // Build x^k - c_0 x^{k-1} - ... - c_{k-1} low-order first: the x^k term is monic (1),
    // and the coefficient of x^(k-1-i) is -c_i, i.e. coeff[k-1-i] = -coeffs[i].
    std::vector<Rational> out(k + 1);
    out[k] = Rational::from_int(1);
    for (std::size_t i = 0; i < k; ++i) {
        auto neg = coeffs[i].negate();
        if (!neg) {
            return make_error<RationalPoly>(neg.error());
        }
        out[k - 1 - i] = *neg;
    }
    return RationalPoly::from_coeffs(std::move(out));
}

auto characteristic_roots(std::span<const Rational> coeffs)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>> {
    using Roots = std::vector<std::pair<Rational, std::int64_t>>;
    auto poly = characteristic_polynomial(coeffs);
    if (!poly) {
        return make_error<Roots>(poly.error());
    }
    return rational_roots(*poly);  // rational roots with multiplicities; monic, non-zero
}

auto all_roots_rational(std::span<const Rational> coeffs) -> Result<bool> {
    auto roots = characteristic_roots(coeffs);
    if (!roots) {
        return make_error<bool>(roots.error());
    }
    // The characteristic polynomial has degree k == coeffs.size(); it splits over Q iff
    // the rational-root multiplicities account for all k of its roots.
    std::int64_t total = 0;
    for (const auto& [root, mult] : *roots) {
        total += mult;  // each mult >= 1, at most k roots, so the sum cannot overflow int64
    }
    return total == static_cast<std::int64_t>(coeffs.size());
}

namespace {

// base^exp for exp >= 0 by repeated exact multiplication (exp is small — at most k-1 in
// this module). base^0 == 1 for every base, including 0 (the 0^0 == 1 convention that
// makes the n = 0 row of the confluent-Vandermonde matrix come out right). Fails only on
// overflow of the exact rational multiply.
[[nodiscard]] auto rat_pow(const Rational& base, std::int64_t exp) -> Result<Rational> {
    assert(exp >= 0 && "rat_pow: negative exponent");
    Rational result = Rational::from_int(1);
    for (std::int64_t i = 0; i < exp; ++i) {
        auto p = result.multiply(base);
        if (!p) {
            return p;
        }
        result = *p;
    }
    return result;
}

// A rational as a canonical constant Expr: an integer collapses to an integer leaf,
// otherwise a reduced fraction. Rational is already canonical (den > 0, never INT64_MIN),
// so Expr::rational cannot fail here, but its Result is threaded through regardless.
[[nodiscard]] auto rational_to_expr(const Rational& r) -> Result<Expr> {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator());
}

// One basis solution of the general closed form: the term n^j * r^n. The flat list of
// these (roots in characteristic_roots order, j ascending within each root) fixes both
// the column order of the linear system and the order the solved constants are consumed.
struct Basis {
    Rational root;
    std::int64_t power;  // the j in n^j
};

}  // namespace

auto closed_form(std::span<const Rational> coeffs, std::span<const Rational> initial)
    -> Result<Expr> {
    if (coeffs.empty()) {
        return make_error<Expr>(MathError::domain_error);  // no recurrence
    }
    if (initial.size() != coeffs.size()) {
        return make_error<Expr>(MathError::domain_error);  // need exactly k initial values
    }
    const std::size_t k = coeffs.size();

    // Only the split-over-Q case has a rational closed form; otherwise be honest.
    auto splits = all_roots_rational(coeffs);
    if (!splits) {
        return make_error<Expr>(splits.error());
    }
    if (!*splits) {
        return make_error<Expr>(MathError::not_implemented);
    }
    auto roots_res = characteristic_roots(coeffs);
    if (!roots_res) {
        return make_error<Expr>(roots_res.error());
    }
    const auto& roots = *roots_res;

    // Flat basis list n^j * r^n; its length equals k because the polynomial splits.
    std::vector<Basis> basis;
    basis.reserve(k);
    for (const auto& [root, mult] : roots) {
        for (std::int64_t j = 0; j < mult; ++j) {
            basis.push_back(Basis{.root = root, .power = j});
        }
    }
    assert(basis.size() == k && "split polynomial: basis size must equal the order");

    // Confluent-Vandermonde system M c = b: row n (n = 0..k-1) is a(n) = a_n, i.e.
    // sum_col c_col * (n^{basis[col].power} * basis[col].root^n) = initial[n].
    std::vector<std::vector<Rational>> m_rows(k, std::vector<Rational>(k));
    std::vector<std::vector<Rational>> b_rows(k, std::vector<Rational>(1));
    for (std::size_t n = 0; n < k; ++n) {
        for (std::size_t col = 0; col < k; ++col) {
            auto rpow = rat_pow(basis[col].root, static_cast<std::int64_t>(n));
            if (!rpow) {
                return make_error<Expr>(rpow.error());
            }
            auto npow = rat_pow(Rational::from_int(static_cast<std::int64_t>(n)),
                                basis[col].power);
            if (!npow) {
                return make_error<Expr>(npow.error());
            }
            auto entry = npow->multiply(*rpow);
            if (!entry) {
                return make_error<Expr>(entry.error());
            }
            m_rows[n][col] = *entry;
        }
        b_rows[n][0] = initial[n];
    }

    auto m = Matrix::from_rows(std::move(m_rows));
    if (!m) {
        return make_error<Expr>(m.error());
    }
    auto b = Matrix::from_rows(std::move(b_rows));
    if (!b) {
        return make_error<Expr>(b.error());
    }
    auto sol = m->solve(*b);  // domain_error if singular (e.g. a repeated zero root)
    if (!sol) {
        return make_error<Expr>(sol.error());
    }

    // Assemble a(n) = sum over roots of (poly in n) * r^n from the solved constants,
    // consuming them in the same (root, j-ascending) order as the basis was built.
    const Expr n_sym = Expr::symbol("n");
    std::vector<Expr> outer_terms;
    outer_terms.reserve(roots.size());
    std::size_t idx = 0;
    for (const auto& [root, mult] : roots) {
        std::vector<Expr> poly_terms;
        poly_terms.reserve(static_cast<std::size_t>(mult));
        for (std::int64_t j = 0; j < mult; ++j) {
            auto c_expr = rational_to_expr(sol->at(idx, 0));
            if (!c_expr) {
                return make_error<Expr>(c_expr.error());
            }
            ++idx;
            if (j == 0) {
                poly_terms.push_back(*c_expr);  // constant term c_{i,0}
            } else {
                // c_{i,j} * n^j
                poly_terms.push_back(
                    Expr::product({*c_expr, Expr::power(n_sym, Expr::integer(j))}));
            }
        }
        Expr poly_part = poly_terms.size() == 1 ? poly_terms.front()
                                                : Expr::sum(std::move(poly_terms));
        auto r_expr = rational_to_expr(root);
        if (!r_expr) {
            return make_error<Expr>(r_expr.error());
        }
        const Expr base_pow = Expr::power(*r_expr, n_sym);  // r^n
        outer_terms.push_back(Expr::product({std::move(poly_part), base_pow}));
    }
    return outer_terms.size() == 1 ? outer_terms.front() : Expr::sum(std::move(outer_terms));
}

auto generating_function(std::span<const Rational> coeffs, std::span<const Rational> initial)
    -> Result<GeneratingFunction> {
    if (coeffs.empty()) {
        return make_error<GeneratingFunction>(MathError::domain_error);  // no recurrence
    }
    if (initial.size() != coeffs.size()) {
        return make_error<GeneratingFunction>(MathError::domain_error);
    }
    const std::size_t k = coeffs.size();

    // Denominator Q(x) = 1 - c_0 x - c_1 x^2 - ... - c_{k-1} x^k (low-order first):
    // q[0] = 1 and q[j] = -c_{j-1} for j = 1..k.
    std::vector<Rational> q(k + 1);
    q[0] = Rational::from_int(1);
    for (std::size_t j = 1; j <= k; ++j) {
        auto neg = coeffs[j - 1].negate();
        if (!neg) {
            return make_error<GeneratingFunction>(neg.error());
        }
        q[j] = *neg;
    }

    // Numerator P(x) = sum_{m=0}^{k-1} p_m x^m with p_m = a_m - sum_{j=1}^{m} c_{j-1} a_{m-j}
    // (the convolution G*Q truncated below degree k, where the recurrence has not yet
    // taken over). Every a index used here is an initial condition (m - j < m <= k-1).
    std::vector<Rational> p(k);
    for (std::size_t m = 0; m < k; ++m) {
        Rational acc = initial[m];  // a_m
        for (std::size_t j = 1; j <= m; ++j) {
            auto term = coeffs[j - 1].multiply(initial[m - j]);  // c_{j-1} * a_{m-j}
            if (!term) {
                return make_error<GeneratingFunction>(term.error());
            }
            auto next = acc.subtract(*term);
            if (!next) {
                return make_error<GeneratingFunction>(next.error());
            }
            acc = *next;
        }
        p[m] = acc;
    }

    return GeneratingFunction{.numerator = RationalPoly::from_coeffs(std::move(p)),
                              .denominator = RationalPoly::from_coeffs(std::move(q))};
}

}  // namespace nimblecas
