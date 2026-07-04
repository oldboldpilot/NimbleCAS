// NimbleCAS unbounded exact integer combinatorics (BigInt-backed).
// @author Olumuyiwa Oluwasanmi
//
// This is the overflow-free counterpart of nimblecas.combinatorics. That module keeps
// every result in an int64 and reports MathError::overflow the instant an exact answer
// would exceed the representable range (factorial past 20, fibonacci past 92, catalan
// from n = 36, and so on). Here every count is carried in a BigInt (see nimblecas.bigint),
// which allocates as many limbs as the exact answer needs, so NONE of these routines can
// overflow. The trade-off is honest: the int64 module is faster (native machine words),
// this one is unbounded (arbitrary precision). A negative argument or an out-of-range k
// is still a MathError::domain_error, matching the semantics of the int64 module
// function-for-function; there is simply no overflow error to return.
//
// Exactness is preserved the same way as in the int64 module: the multiplicative
// binomial/catalan/multinomial formulas interleave an EXACT division after each
// multiplication, so the running value is always an integer. Because BigInt::divmod is
// the only division available, each such divide is performed as a divmod and the
// remainder is asserted to be zero (the quotient is provably exact at that step).
//
// Scope: this module covers the INTEGER-valued sequences. Rational-valued sequences such
// as the Bernoulli numbers are deliberately absent — they need a BigRational-backed
// module, not this one; the int64 nimblecas.combinatorics still offers bernoulli() over
// exact (bounded) Rational.

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.bigcombinatorics;

import std;
import nimblecas.core;
import nimblecas.bigint;

export namespace nimblecas {

// --- Unbounded exact integer sequences (BigInt, never overflow) -------------

// n! for n >= 0, as the iterative product 1 * 2 * ... * n (0! = 1). Fails with
// domain_error for n < 0. Unlike the int64 factorial there is no overflow ceiling: the
// magnitude simply grows (25! and 50! are exact here, whereas 21! already overflows int64).
[[nodiscard]] auto factorial(std::int64_t n) -> Result<BigInt>;

// The binomial coefficient C(n, k) = n! / (k! (n-k)!). Returns 0 when k < 0 or k > n
// (the conventional empty count, matching the int64 module); fails with domain_error for
// n < 0. Uses the multiplicative formula prod_{i=1..k} (n-k+i) / i with an EXACT divide
// after each multiply, so the running value is an exact integer that never exceeds the
// final answer — never the wasteful factorial/factorial/factorial.
[[nodiscard]] auto binomial(std::int64_t n, std::int64_t k) -> Result<BigInt>;

// The multinomial coefficient (sum k_i)! / prod(k_i!) — the number of distinct orderings
// of a multiset with the given multiplicities. All k_i must be >= 0, else domain_error.
// An empty span yields 1 (the empty product). Each factorial division is exact.
[[nodiscard]] auto multinomial(std::span<const std::int64_t> ks) -> Result<BigInt>;

// The n-th Catalan number C_n = C(2n, n) / (n + 1), exact. domain_error for n < 0. Where
// the int64 catalan overflows at n = 36, this one keeps counting.
[[nodiscard]] auto catalan(std::int64_t n) -> Result<BigInt>;

// The falling factorial n^(k) = n (n-1) ... (n-k+1), a product of exactly k descending
// factors (the empty product 1 when k = 0). Requires k >= 0 (else domain_error); n is
// unrestricted, so a negative n simply yields a signed result.
[[nodiscard]] auto falling_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt>;

// The rising factorial n^(k)- = n (n+1) ... (n+k-1), a product of exactly k ascending
// factors (the empty product 1 when k = 0). Requires k >= 0 (else domain_error); n is
// unrestricted.
[[nodiscard]] auto rising_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt>;

// The double factorial n!! = n (n-2) (n-4) ... down to 1 or 2, with the empty-product
// convention (-1)!! = 0!! = 1. Requires n >= -1 (else domain_error).
[[nodiscard]] auto double_factorial(std::int64_t n) -> Result<BigInt>;

// The n-th Fibonacci number (F_0 = 0, F_1 = 1), via the FAST-DOUBLING identities
// F(2k) = F(k) (2 F(k+1) - F(k)) and F(2k+1) = F(k+1)^2 + F(k)^2 — O(log n) BigInt
// multiplies rather than O(n) additions. domain_error for n < 0. F(93) and beyond
// (which overflow int64) are exact here.
[[nodiscard]] auto fibonacci(std::int64_t n) -> Result<BigInt>;

// The n-th Lucas number (L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}). domain_error for
// n < 0. Unbounded.
[[nodiscard]] auto lucas(std::int64_t n) -> Result<BigInt>;

// Stirling number of the second kind S(n, k): the number of ways to partition a set of n
// labelled elements into k non-empty unlabelled subsets, via the recurrence
// S(n, k) = k S(n-1, k) + S(n-1, k-1). Returns 0 when k > n; domain_error for n < 0 or
// k < 0. Exact and unbounded.
[[nodiscard]] auto stirling_second(std::int64_t n, std::int64_t k) -> Result<BigInt>;

// Unsigned Stirling number of the first kind c(n, k): the number of permutations of n
// elements with exactly k disjoint cycles, via c(n, k) = (n-1) c(n-1, k) + c(n-1, k-1).
// Returns 0 when k > n; domain_error for n < 0 or k < 0. Exact and unbounded.
[[nodiscard]] auto stirling_first_unsigned(std::int64_t n, std::int64_t k) -> Result<BigInt>;

// The n-th Bell number B_n: the number of partitions of a set of n elements, computed via
// the Bell triangle. domain_error for n < 0. Exact and unbounded.
[[nodiscard]] auto bell(std::int64_t n) -> Result<BigInt>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Divide a by b when the quotient is known a priori to be exact (zero remainder). Division
// by zero is propagated. Exactness holds by construction at every call site here, but the
// remainder is checked on the RAILWAY (not merely asserted) so that a would-be inexact
// division surfaces as undefined_value instead of silently truncating under -DNDEBUG.
[[nodiscard]] auto exact_div(const BigInt& a, const BigInt& b) -> Result<BigInt> {
    auto dm = a.divmod(b);
    if (!dm) {
        return make_error<BigInt>(dm.error());
    }
    if (!dm->second.is_zero()) {
        return make_error<BigInt>(MathError::undefined_value);  // exactness invariant violated
    }
    return std::move(dm->first);
}

}  // namespace

auto factorial(std::int64_t n) -> Result<BigInt> {
    if (n < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    BigInt result = BigInt::from_u64(1);
    for (std::int64_t i = 2; i <= n; ++i) {
        result = result.multiply(BigInt::from_i64(i));
    }
    return result;
}

auto binomial(std::int64_t n, std::int64_t k) -> Result<BigInt> {
    if (n < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    if (k < 0 || k > n) {
        return BigInt{};  // the conventional empty count (matches the int64 module)
    }
    // C(n, k) == C(n, n-k); pick the smaller upper index to minimise the loop count.
    if (k > n - k) {
        k = n - k;
    }
    // Running value C(n-k+i, i) is an exact integer at every step, so each divide is exact
    // and the magnitude never exceeds the final answer.
    BigInt result = BigInt::from_u64(1);
    for (std::int64_t i = 1; i <= k; ++i) {
        result = result.multiply(BigInt::from_i64(n - k + i));
        auto q = exact_div(result, BigInt::from_i64(i));
        if (!q) {
            return q;
        }
        result = std::move(*q);
    }
    return result;
}

auto multinomial(std::span<const std::int64_t> ks) -> Result<BigInt> {
    std::int64_t total = 0;
    for (const std::int64_t k : ks) {
        if (k < 0) {
            return make_error<BigInt>(MathError::domain_error);
        }
        if (total > std::numeric_limits<std::int64_t>::max() - k) {
            return make_error<BigInt>(MathError::domain_error);  // sum of k_i overflows int64
        }
        total += k;
    }
    // (sum k_i)! / prod(k_i!). Every partial quotient S!/(k_1! ... k_j!) is an integer, so
    // dividing by one factorial at a time keeps the running value exact throughout.
    auto num = factorial(total);
    if (!num) {
        return num;
    }
    BigInt result = std::move(*num);
    for (const std::int64_t k : ks) {
        auto denom = factorial(k);
        if (!denom) {
            return denom;
        }
        auto q = exact_div(result, *denom);
        if (!q) {
            return q;
        }
        result = std::move(*q);
    }
    return result;
}

auto catalan(std::int64_t n) -> Result<BigInt> {
    if (n < 0 || n > std::numeric_limits<std::int64_t>::max() / 2 - 1) {
        return make_error<BigInt>(MathError::domain_error);  // n<0, or 2*n / n+1 overflows int64
    }
    // C_n = C(2n, n) / (n + 1), exact.
    auto central = binomial(2 * n, n);
    if (!central) {
        return central;
    }
    return exact_div(*central, BigInt::from_i64(n + 1));
}

auto falling_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt> {
    if (k < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    // n (n-1) ... (n-k+1): exactly k descending factors (empty product 1 when k == 0).
    BigInt result = BigInt::from_u64(1);
    for (std::int64_t i = 0; i < k; ++i) {
        result = result.multiply(BigInt::from_i64(n - i));
    }
    return result;
}

auto rising_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt> {
    if (k < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    // n (n+1) ... (n+k-1): exactly k ascending factors (empty product 1 when k == 0).
    BigInt result = BigInt::from_u64(1);
    for (std::int64_t i = 0; i < k; ++i) {
        result = result.multiply(BigInt::from_i64(n + i));
    }
    return result;
}

auto double_factorial(std::int64_t n) -> Result<BigInt> {
    if (n < -1) {
        return make_error<BigInt>(MathError::domain_error);
    }
    // (-1)!! = 0!! = 1 (empty product); otherwise step down by two to 1 (odd) or 2 (even).
    BigInt result = BigInt::from_u64(1);
    for (std::int64_t i = n; i >= 2; i -= 2) {
        result = result.multiply(BigInt::from_i64(i));
    }
    return result;
}

auto fibonacci(std::int64_t n) -> Result<BigInt> {
    if (n < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    // Fast doubling, scanning the bits of n from most significant to least. The pair
    // (a, b) tracks (F(k), F(k+1)); each processed bit doubles k (and adds the bit).
    const auto un = static_cast<std::uint64_t>(n);
    BigInt a = BigInt{};             // F(0) = 0
    BigInt b = BigInt::from_u64(1);  // F(1) = 1
    for (int i = 63 - std::countl_zero(un); i >= 0; --i) {
        // F(2k)   = F(k) * (2 F(k+1) - F(k))
        const BigInt two_b_minus_a = b.add(b).subtract(a);
        const BigInt c = a.multiply(two_b_minus_a);
        // F(2k+1) = F(k+1)^2 + F(k)^2
        const BigInt d = a.multiply(a).add(b.multiply(b));
        if (((un >> i) & 1U) != 0) {
            a = d;             // F(2k+1)
            b = c.add(d);      // F(2k+2) = F(2k) + F(2k+1)
        } else {
            a = c;             // F(2k)
            b = d;             // F(2k+1)
        }
    }
    return a;
}

auto lucas(std::int64_t n) -> Result<BigInt> {
    if (n < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    if (n == 0) {
        return BigInt::from_u64(2);  // L_0 = 2
    }
    BigInt prev = BigInt::from_u64(2);  // L_0
    BigInt curr = BigInt::from_u64(1);  // L_1
    for (std::int64_t i = 2; i <= n; ++i) {
        BigInt next = prev.add(curr);
        prev = std::move(curr);
        curr = std::move(next);
    }
    return curr;
}

auto stirling_second(std::int64_t n, std::int64_t k) -> Result<BigInt> {
    if (n < 0 || k < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    if (k > n) {
        return BigInt{};
    }
    // S(i, j) = j S(i-1, j) + S(i-1, j-1), one row updated in descending j so each read
    // sees the previous row's value.
    std::vector<BigInt> dp(static_cast<std::size_t>(k) + 1);
    dp[0] = BigInt::from_u64(1);  // S(0, 0) = 1
    for (std::int64_t i = 1; i <= n; ++i) {
        const std::int64_t top = std::min(i, k);
        for (std::int64_t j = top; j >= 1; --j) {
            const auto u = static_cast<std::size_t>(j);
            dp[u] = BigInt::from_i64(j).multiply(dp[u]).add(dp[u - 1]);
        }
        dp[0] = BigInt{};  // S(i, 0) = 0 for i >= 1
    }
    return dp[static_cast<std::size_t>(k)];
}

auto stirling_first_unsigned(std::int64_t n, std::int64_t k) -> Result<BigInt> {
    if (n < 0 || k < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    if (k > n) {
        return BigInt{};
    }
    // c(i, j) = c(i-1, j-1) + (i-1) c(i-1, j), same descending-j single-row update.
    std::vector<BigInt> dp(static_cast<std::size_t>(k) + 1);
    dp[0] = BigInt::from_u64(1);  // c(0, 0) = 1
    for (std::int64_t i = 1; i <= n; ++i) {
        const std::int64_t top = std::min(i, k);
        for (std::int64_t j = top; j >= 1; --j) {
            const auto u = static_cast<std::size_t>(j);
            dp[u] = BigInt::from_i64(i - 1).multiply(dp[u]).add(dp[u - 1]);
        }
        dp[0] = BigInt{};  // c(i, 0) = 0 for i >= 1
    }
    return dp[static_cast<std::size_t>(k)];
}

auto bell(std::int64_t n) -> Result<BigInt> {
    if (n < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    // Bell triangle: row 0 is [1]; each new row starts with the previous row's last entry,
    // and every subsequent entry is the sum of the entry to its left and the entry above
    // it. B_n is the first (leftmost) entry of row n.
    std::vector<BigInt> row;
    row.push_back(BigInt::from_u64(1));  // row 0 = [1], B_0 = 1
    for (std::int64_t i = 1; i <= n; ++i) {
        std::vector<BigInt> next(static_cast<std::size_t>(i) + 1);
        next[0] = row.back();  // start with the previous row's last entry
        for (std::int64_t j = 1; j <= i; ++j) {
            const auto u = static_cast<std::size_t>(j);
            next[u] = next[u - 1].add(row[u - 1]);
        }
        row = std::move(next);
    }
    return row[0];
}

}  // namespace nimblecas
