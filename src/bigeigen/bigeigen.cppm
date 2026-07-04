// NimbleCAS exact characteristic polynomials and rational eigenvalues over BigRational.
// @author Olumuyiwa Oluwasanmi
//
// This module answers two exact linear-algebra questions about a square BigMatrix over Q,
// with no rounding and no overflow ceiling (BigRational is the unbounded exact tier — see
// nimblecas.bigrational):
//
//   1. characteristic_polynomial(A) — the coefficients of p(x) = det(xI - A), computed
//      EXACTLY via the FADDEEV-LEVERRIER algorithm. That algorithm is the right fit for the
//      big tier because every step it needs is one BigMatrix/BigRational already supports
//      exactly: a matrix multiply, a trace (a sum of diagonal entries), and a division by a
//      small positive integer k in {1, ..., n}. No eigen-solver, no floating point, no
//      characteristic-matrix symbolic expansion — just the field operations of Q, each of
//      which BigRational performs without overflow. The returned polynomial is MONIC and
//      given LOW-DEGREE-FIRST (coefficient of x^j at index j), matching the coefficient
//      convention used across the engine (e.g. nimblecas.roots / RationalPoly).
//
//   2. rational_eigenvalues(A) — the RATIONAL eigenvalues with multiplicity, i.e. the
//      rational roots of that characteristic polynomial, found via the RATIONAL ROOT THEOREM
//      + deflation, implemented directly over BigInt/BigRational so it never overflows.
//
// Two cheap, exact byproducts of the same Faddeev-LeVerrier sweep are also exposed:
// determinant(A) = (-1)^n * p(0), and inverse(A) = -M_n / c_n (the last Faddeev matrix
// scaled by the reciprocal of the constant term), which fails with domain_error when A is
// singular.
//
// ---------------------------------------------------------------------------------------
// HONESTY BOUNDARY (true in behaviour, not just in this comment):
//   * The characteristic polynomial IS the full, exact answer: every eigenvalue (rational,
//     irrational, or complex) is a root of it, and all n+1 coefficients are exact elements
//     of Q.
//   * rational_eigenvalues returns ONLY the RATIONAL eigenvalues. Irrational eigenvalues
//     (surds, e.g. the +/-sqrt(2) of [[0,1],[2,0]]) and complex eigenvalues (e.g. the +/-i
//     of the rotation [[0,1],[-1,0]]) are NOT extracted — no radicals, no RootOf, no complex
//     field. Such an eigenvalue simply does not appear in the returned list. The caller can
//     always tell whether the spectrum was fully rational by summing the returned
//     multiplicities and comparing against n: a total below n means some eigenvalues are
//     irrational or complex and were (honestly) omitted. This is NOT a full eigendecomposition.
//   * Divisor enumeration inside the rational-root search is trial division over BigInt, so
//     it is exact but its cost grows with the size of the constant / leading coefficient's
//     prime factors. It is fast when those are small (the typical CAS case); a characteristic
//     polynomial whose constant term is a large semiprime is a pathological slow case, not an
//     incorrect one.

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.bigeigen;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigmatrix;

export namespace nimblecas {

// The exact characteristic polynomial p(x) = det(xI - A), as its BigRational coefficients
// LOW-DEGREE-FIRST: the returned vector r has size n + 1 with r[j] the coefficient of x^j,
// and is always MONIC (r[n] == 1). Computed by Faddeev-LeVerrier (see the module header).
// A non-square matrix yields MathError::domain_error; the 0x0 matrix yields the constant
// polynomial {1} (its empty-product characteristic polynomial).
[[nodiscard]] auto characteristic_polynomial(const BigMatrix& a)
    -> Result<std::vector<BigRational>>;

// The exact determinant as the Faddeev-LeVerrier byproduct det(A) = (-1)^n * p(0), where
// p(0) is the constant term of the characteristic polynomial. Agrees with
// BigMatrix::determinant (which uses fraction-free Bareiss); both are exact. Non-square
// input yields domain_error; the 0x0 determinant is the empty product 1.
[[nodiscard]] auto determinant(const BigMatrix& a) -> Result<BigRational>;

// The exact inverse as the Faddeev-LeVerrier byproduct A^{-1} = -M_n / c_n, where M_n is
// the final Faddeev matrix and c_n = p(0) is the constant term. A SINGULAR matrix (c_n == 0,
// equivalently det(A) == 0) has no inverse and yields MathError::domain_error; a non-square
// matrix likewise yields domain_error. The 0x0 matrix inverts to the 0x0 matrix.
[[nodiscard]] auto inverse(const BigMatrix& a) -> Result<BigMatrix>;

// The RATIONAL eigenvalues of A, each paired with its multiplicity (>= 1), sorted ascending
// by eigenvalue. These are exactly the rational roots of the characteristic polynomial, via
// the rational root theorem + deflation over BigInt/BigRational. See the module header's
// HONESTY BOUNDARY: irrational and complex eigenvalues are NOT returned, so the sum of the
// reported multiplicities is n iff the spectrum is entirely rational (and is < n otherwise).
// A non-square matrix yields MathError::domain_error; the 0x0 matrix yields an empty list.
[[nodiscard]] auto rational_eigenvalues(const BigMatrix& a)
    -> Result<std::vector<std::pair<BigRational, std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Sum of the diagonal entries of m (trace). For the square matrices used here rows == cols;
// the std::min guard keeps it well defined for any shape. BigRational add is infallible, so
// this cannot fail.
[[nodiscard]] auto matrix_trace(const BigMatrix& m) -> BigRational {
    BigRational acc{};  // 0/1
    const std::size_t d = std::min(m.rows(), m.cols());
    for (std::size_t i = 0; i < d; ++i) {
        acc = acc.add(m.at(i, i));
    }
    return acc;
}

// The full Faddeev-LeVerrier state: the characteristic-polynomial coefficients (low-degree
// first, monic) and the final Faddeev matrix M_n (needed for the inverse byproduct).
struct FaddeevState {
    std::vector<BigRational> coeffs;  // p(x) low-degree-first, size n + 1, coeffs[n] == 1
    BigMatrix m_final;                // M_n, satisfying A * M_n + coeffs[0] * I == 0
    std::size_t n{};                  // dimension
};

// Run Faddeev-LeVerrier on a square matrix a. The recurrence (with M_1 = I, c_0 = 1) is
//   M_k     : current Faddeev matrix (starts at M_1 = I)
//   c_k     = -trace(A * M_k) / k
//   M_{k+1} = A * M_k + c_k * I
// and p(x) = x^n + c_1 x^{n-1} + ... + c_n = sum_{k=0}^{n} c_k x^{n-k} with c_0 = 1. Only one
// matrix product per iteration is needed: A * M_k, computed for the trace, is exactly the
// term reused to form M_{k+1}, so we never multiply twice. Every division is by an integer
// k in {1, ..., n} (never zero) and is exact over Q.
[[nodiscard]] auto run_faddeev_leverrier(const BigMatrix& a) -> Result<FaddeevState> {
    if (!a.is_square()) {
        return make_error<FaddeevState>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // c_desc[k] = c_k (descending powers), with the leading coefficient c_0 = 1.
    std::vector<BigRational> c_desc(n + 1);
    c_desc[0] = BigRational::from_int(1);

    // M starts as M_1 = I. For n == 0 the loop below never runs and M stays the (empty) 0x0
    // identity, which is the correct M_0 for the degenerate case.
    BigMatrix m = BigMatrix::identity(n);
    for (std::size_t k = 1; k <= n; ++k) {
        // A * M_k, reused both for the trace and (when k < n) to build M_{k+1}.
        auto am = a.multiply(m);
        if (!am) {
            return make_error<FaddeevState>(am.error());
        }
        // c_k = -trace(A * M_k) / k. k >= 1, so the divisor is nonzero and the divide is exact.
        const BigRational tr = matrix_trace(*am);
        auto ck = tr.negate().divide(BigRational::from_int(static_cast<std::int64_t>(k)));
        if (!ck) {
            return make_error<FaddeevState>(ck.error());
        }
        c_desc[k] = *ck;

        if (k < n) {
            // M_{k+1} = A * M_k + c_k * I.
            auto scaled_id = BigMatrix::identity(n).scale(*ck);
            if (!scaled_id) {
                return make_error<FaddeevState>(scaled_id.error());
            }
            auto next = am->add(*scaled_id);
            if (!next) {
                return make_error<FaddeevState>(next.error());
            }
            m = std::move(*next);  // m is now M_{k+1}
        }
        // On the final iteration (k == n) m is left as M_n, exactly what the inverse needs.
    }

    // Reindex descending c_desc into a low-degree-first coefficient vector:
    //   p(x) = sum_{k=0}^{n} c_desc[k] x^{n-k}  =>  coeff of x^j is c_desc[n - j].
    std::vector<BigRational> coeffs(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        coeffs[j] = c_desc[n - j];
    }
    return FaddeevState{std::move(coeffs), std::move(m), n};
}

// --- Rational-root machinery over BigInt/BigRational ------------------------

// Clear denominators: return the integer coefficients (BigInt, low-degree-first) of the
// polynomial obtained by multiplying p by the LCM of all its denominators. p must be
// non-empty. Each intermediate division is exact (the divisor divides its dividend) and
// unbounded, so this never overflows and only threads Result defensively.
[[nodiscard]] auto clear_denominators(const std::vector<BigRational>& p)
    -> Result<std::vector<BigInt>> {
    // lcm of all (positive) denominators: lcm <- lcm / gcd(lcm, d) * d.
    BigInt lcm = BigInt::from_u64(1);
    for (const BigRational& c : p) {
        const BigInt& d = c.denominator();  // canonical: d > 0
        const BigInt g = BigInt::gcd(lcm, d);  // g >= 1, divides lcm
        auto reduced = lcm.divide(g);
        if (!reduced) {
            return make_error<std::vector<BigInt>>(reduced.error());
        }
        lcm = reduced->multiply(d);
    }
    std::vector<BigInt> out;
    out.reserve(p.size());
    for (const BigRational& c : p) {
        auto factor = lcm.divide(c.denominator());  // exact: denominator | lcm
        if (!factor) {
            return make_error<std::vector<BigInt>>(factor.error());
        }
        out.push_back(c.numerator().multiply(*factor));
    }
    return out;
}

// The positive divisors of |n| (n != 0), unsorted and without duplicates, via trial division
// up to sqrt(|n|). Exact but O(sqrt(|n|)) BigInt operations (see the module HONESTY note).
[[nodiscard]] auto bigint_divisors(const BigInt& n) -> Result<std::vector<BigInt>> {
    const BigInt a = n.abs();
    assert(!a.is_zero() && "bigint_divisors requires a non-zero argument");
    const BigInt one = BigInt::from_u64(1);
    std::vector<BigInt> ds;
    // Loop while i*i <= a. i*i is unbounded (no overflow), so the bound is exact.
    for (BigInt i = one; i.multiply(i) <= a; i = i.add(one)) {
        auto dm = a.divmod(i);  // i >= 1, never zero
        if (!dm) {
            return make_error<std::vector<BigInt>>(dm.error());
        }
        if (dm->second.is_zero()) {
            ds.push_back(i);
            const BigInt& other = dm->first;  // a / i
            if (!(other == i)) {
                ds.push_back(other);
            }
        }
    }
    return ds;
}

// Synthetic division of a low-degree-first polynomial p (degree >= 1) by (x - r). Returns the
// quotient (low-degree-first, one degree lower) and the scalar remainder, which is exactly
// p(r) — so a zero remainder both confirms r as a root and yields the deflated polynomial in
// one pass. All arithmetic is infallible BigRational.
[[nodiscard]] auto deflate(const std::vector<BigRational>& p, const BigRational& r)
    -> std::pair<std::vector<BigRational>, BigRational> {
    const std::size_t d = p.size() - 1;  // degree (caller guarantees d >= 1)
    std::vector<BigRational> q(d);        // quotient has d coefficients (degree d - 1)
    // High-to-low synthetic division: q_{d-1} = p_d, q_{k-1} = p_k + r * q_k.
    BigRational carry = p[d];
    q[d - 1] = carry;
    for (std::size_t k = d; k-- > 1;) {  // k = d-1, d-2, ..., 1
        carry = p[k].add(r.multiply(carry));
        q[k - 1] = carry;
    }
    const BigRational remainder = p[0].add(r.multiply(carry));  // == p(r)
    return {std::move(q), remainder};
}

}  // namespace

// --- Public API -------------------------------------------------------------

auto characteristic_polynomial(const BigMatrix& a) -> Result<std::vector<BigRational>> {
    auto state = run_faddeev_leverrier(a);
    if (!state) {
        return make_error<std::vector<BigRational>>(state.error());
    }
    return std::move(state->coeffs);
}

auto determinant(const BigMatrix& a) -> Result<BigRational> {
    auto state = run_faddeev_leverrier(a);
    if (!state) {
        return make_error<BigRational>(state.error());
    }
    // det(A) = (-1)^n * p(0); p(0) is the constant term (low-degree-first index 0).
    const BigRational& constant_term = state->coeffs.front();  // coeffs is size n+1 >= 1
    return (state->n % 2 == 0) ? constant_term : constant_term.negate();
}

auto inverse(const BigMatrix& a) -> Result<BigMatrix> {
    auto state = run_faddeev_leverrier(a);
    if (!state) {
        return make_error<BigMatrix>(state.error());
    }
    // A^{-1} = -M_n / c_n, where c_n = p(0) is the constant term. c_n == 0 <=> det(A) == 0
    // <=> A is singular, which has no inverse.
    const BigRational& c_n = state->coeffs.front();
    if (c_n.is_zero()) {
        return make_error<BigMatrix>(MathError::domain_error);  // singular
    }
    auto reciprocal = c_n.reciprocal();  // c_n != 0, so this succeeds
    if (!reciprocal) {
        return make_error<BigMatrix>(reciprocal.error());
    }
    // factor = -1 / c_n; scaling M_n by it yields the inverse.
    const BigRational factor = reciprocal->negate();
    return state->m_final.scale(factor);
}

auto rational_eigenvalues(const BigMatrix& a)
    -> Result<std::vector<std::pair<BigRational, std::int64_t>>> {
    using Eigenvalues = std::vector<std::pair<BigRational, std::int64_t>>;

    auto poly = characteristic_polynomial(a);
    if (!poly) {
        return make_error<Eigenvalues>(poly.error());
    }
    const std::vector<BigRational>& p = *poly;
    // p has size n + 1. Degree 0 (the 0x0 matrix, p == {1}) has no eigenvalues.
    if (p.size() <= 1) {
        return Eigenvalues{};
    }

    // Integer coefficients (low-degree-first) for the rational root theorem: any rational
    // root written u/v in lowest terms has u | (lowest nonzero coeff) and v | (leading coeff).
    auto ints = clear_denominators(p);
    if (!ints) {
        return make_error<Eigenvalues>(ints.error());
    }
    const std::vector<BigInt>& b = *ints;

    // Lowest nonzero coefficient supplies the numerator candidates. A zero constant term
    // (b[0] == 0) means 0 is an eigenvalue; it is added as an explicit candidate below.
    std::size_t low = 0;
    while (low < b.size() && b[low].is_zero()) {
        ++low;
    }
    // The characteristic polynomial is monic, so the leading coefficient (and hence b.back())
    // is nonzero; `low` therefore always stops before the end.
    assert(low < b.size() && "monic characteristic polynomial has a nonzero leading term");

    auto numerators = bigint_divisors(b[low]);
    if (!numerators) {
        return make_error<Eigenvalues>(numerators.error());
    }
    auto denominators = bigint_divisors(b.back());
    if (!denominators) {
        return make_error<Eigenvalues>(denominators.error());
    }

    // Assemble the candidate set: 0 (when the constant term vanished) plus every +/- u/v.
    std::vector<BigRational> candidates;
    if (low > 0) {
        candidates.push_back(BigRational{});  // 0
    }
    for (const BigInt& u : *numerators) {
        for (const BigInt& v : *denominators) {
            auto pos = BigRational::make(u, v);  // v >= 1, never zero
            if (!pos) {
                return make_error<Eigenvalues>(pos.error());
            }
            candidates.push_back(*pos);
            candidates.push_back(pos->negate());
        }
    }

    // Test each candidate on the shrinking work polynomial and deflate confirmed roots to
    // recover multiplicity. Deflating on `work` (rather than the original) makes duplicate
    // candidates (e.g. 2/1 and 4/2) harmless: once a root is fully removed it no longer
    // divides `work`, so a later duplicate contributes nothing.
    Eigenvalues eigenvalues;
    std::vector<BigRational> work = p;  // monic; stays monic under deflation by (x - r)
    for (const BigRational& r : candidates) {
        std::int64_t mult = 0;
        while (work.size() >= 2) {  // degree >= 1: deflation by (x - r) is defined
            auto [quotient, remainder] = deflate(work, r);
            if (!remainder.is_zero()) {
                break;  // r is not a root of the current work polynomial
            }
            ++mult;
            work = std::move(quotient);
        }
        if (mult > 0) {
            eigenvalues.emplace_back(r, mult);
        }
    }

    // Deterministic ascending order by eigenvalue (BigRational is a total order).
    std::ranges::sort(eigenvalues,
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return eigenvalues;
}

}  // namespace nimblecas
