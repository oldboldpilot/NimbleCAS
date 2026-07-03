// NimbleCAS combinatorics and integer sequences (ROADMAP 7.18).
// @author Olumuyiwa Oluwasanmi
//
// Exact, overflow-checked counting functions over int64, plus Bernoulli numbers as
// exact reduced Rationals. Following the rest of the engine, arithmetic is exact and
// overflow-checked (Rule 32): when an int64 result would exceed the representable range
// the operation returns MathError::overflow rather than silently wrapping, and a
// mathematically undefined argument (e.g. a negative factorial) returns
// MathError::domain_error. No exceptions are thrown; every fallible entry point returns
// Result.
//
// The integer routines are written to keep intermediate magnitudes as small as the exact
// answer allows: binomial and catalan interleave a division after every multiplication
// so the running value is always an exact integer, which both avoids spurious overflow
// and keeps every step within int64 whenever the final answer is.
//
// Bernoulli numbers are computed with the Akiyama-Tanigawa algorithm over exact
// Rational. That recurrence natively yields the "second" Bernoulli numbers with
// B_1 = +1/2; every other B_n is convention-independent. We adopt the more common
// "first" convention B_1 = -1/2 by flipping the sign of that single value (see
// bernoulli()).

export module nimblecas.combinatorics;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// --- Integer sequences (all overflow-checked, exact int64) ------------------

// n! for n >= 0. Fails with domain_error for n < 0 and overflow for n > 20
// (21! exceeds INT64_MAX).
[[nodiscard]] auto factorial(std::int64_t n) -> Result<std::int64_t>;

// The binomial coefficient C(n, k) = n! / (k! (n-k)!). Returns 0 when k < 0 or k > n
// (the conventional empty count); fails with domain_error for n < 0 and overflow if the
// exact value exceeds INT64_MAX. Uses the multiplicative formula with a division after
// every multiplication, so the running value never exceeds the final answer.
[[nodiscard]] auto binomial(std::int64_t n, std::int64_t k) -> Result<std::int64_t>;

// The number of ordered k-arrangements of n items, P(n, k) = n! / (n-k)! (the falling
// factorial n (n-1) ... (n-k+1)). Returns 0 when k < 0 or k > n; domain_error for
// n < 0; overflow if the exact value exceeds INT64_MAX.
[[nodiscard]] auto permutations(std::int64_t n, std::int64_t k) -> Result<std::int64_t>;

// The n-th Catalan number C_n = C(2n, n) / (n + 1). domain_error for n < 0; overflow
// once the value exceeds INT64_MAX (n >= 36).
[[nodiscard]] auto catalan(std::int64_t n) -> Result<std::int64_t>;

// The n-th Fibonacci number (F_0 = 0, F_1 = 1), computed iteratively. domain_error for
// n < 0; overflow once the value exceeds INT64_MAX (n >= 93).
[[nodiscard]] auto fibonacci(std::int64_t n) -> Result<std::int64_t>;

// Stirling number of the second kind S(n, k): the number of ways to partition a set of
// n labelled elements into k non-empty unlabelled subsets. Returns 0 when k > n;
// domain_error for n < 0 or k < 0; overflow if the value exceeds INT64_MAX.
[[nodiscard]] auto stirling_second(std::int64_t n, std::int64_t k) -> Result<std::int64_t>;

// Unsigned Stirling number of the first kind c(n, k): the number of permutations of n
// elements having exactly k disjoint cycles. Returns 0 when k > n; domain_error for
// n < 0 or k < 0; overflow if the value exceeds INT64_MAX.
[[nodiscard]] auto stirling_first(std::int64_t n, std::int64_t k) -> Result<std::int64_t>;

// The n-th Bernoulli number as an exact reduced fraction, using the convention
// B_1 = -1/2 (so B_0 = 1, B_1 = -1/2, B_2 = 1/6, and B_{2k+1} = 0 for k >= 1).
// domain_error for n < 0; overflow if any intermediate Rational exceeds int64.
[[nodiscard]] auto bernoulli(std::int64_t n) -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto add_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_add_overflow(a, b, &out);
}
[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}

}  // namespace

auto factorial(std::int64_t n) -> Result<std::int64_t> {
    if (n < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    std::int64_t result = 1;
    for (std::int64_t i = 2; i <= n; ++i) {
        if (mul_ov(result, i, result)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return result;
}

auto binomial(std::int64_t n, std::int64_t k) -> Result<std::int64_t> {
    if (n < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (k < 0 || k > n) {
        return std::int64_t{0};
    }
    // C(n, k) == C(n, n-k); pick the smaller upper index to minimise the loop count.
    if (k > n - k) {
        k = n - k;
    }
    // Running value P_i = C(n-k+i, i) is an exact integer at every step, so the divide
    // is exact and the value never exceeds the final C(n, k).
    std::int64_t result = 1;
    for (std::int64_t i = 1; i <= k; ++i) {
        if (mul_ov(result, n - k + i, result)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        result /= i;  // exact: P_{i-1} * (n-k+i) is divisible by i
    }
    return result;
}

auto permutations(std::int64_t n, std::int64_t k) -> Result<std::int64_t> {
    if (n < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (k < 0 || k > n) {
        return std::int64_t{0};
    }
    // Falling factorial n (n-1) ... (n-k+1): exactly k descending factors.
    std::int64_t result = 1;
    for (std::int64_t i = 0; i < k; ++i) {
        if (mul_ov(result, n - i, result)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return result;
}

auto catalan(std::int64_t n) -> Result<std::int64_t> {
    if (n < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    // C_i = C_{i-1} * 2 * (2i - 1) / (i + 1); each C_i is an exact integer, so the
    // division is exact and the running value tracks the Catalan numbers themselves
    // rather than the much larger central binomial coefficient C(2n, n).
    std::int64_t result = 1;  // C_0
    for (std::int64_t i = 1; i <= n; ++i) {
        std::int64_t prod = 0;
        if (mul_ov(result, 2, prod) || mul_ov(prod, 2 * i - 1, prod)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        result = prod / (i + 1);  // exact: C_{i-1} * 2 * (2i-1) is divisible by i+1
    }
    return result;
}

auto fibonacci(std::int64_t n) -> Result<std::int64_t> {
    if (n < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (n == 0) {
        return std::int64_t{0};
    }
    std::int64_t prev = 0;  // F_0
    std::int64_t curr = 1;  // F_1
    for (std::int64_t i = 2; i <= n; ++i) {
        std::int64_t next = 0;
        if (add_ov(prev, curr, next)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        prev = curr;
        curr = next;
    }
    return curr;
}

auto stirling_second(std::int64_t n, std::int64_t k) -> Result<std::int64_t> {
    if (n < 0 || k < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (k > n) {
        return std::int64_t{0};
    }
    // Recurrence S(i, j) = j * S(i-1, j) + S(i-1, j-1) over a single row updated in
    // descending j so each read sees the previous row's value.
    std::vector<std::int64_t> dp(static_cast<std::size_t>(k) + 1, 0);
    dp[0] = 1;  // S(0, 0) = 1
    for (std::int64_t i = 1; i <= n; ++i) {
        const std::int64_t top = std::min(i, k);
        for (std::int64_t j = top; j >= 1; --j) {
            const auto u = static_cast<std::size_t>(j);
            std::int64_t term = 0;
            if (mul_ov(j, dp[u], term) || add_ov(term, dp[u - 1], dp[u])) {
                return make_error<std::int64_t>(MathError::overflow);
            }
        }
        dp[0] = 0;  // S(i, 0) = 0 for i >= 1
    }
    return dp[static_cast<std::size_t>(k)];
}

auto stirling_first(std::int64_t n, std::int64_t k) -> Result<std::int64_t> {
    if (n < 0 || k < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (k > n) {
        return std::int64_t{0};
    }
    // Unsigned recurrence c(i, j) = c(i-1, j-1) + (i-1) * c(i-1, j), same descending-j
    // single-row update as above.
    std::vector<std::int64_t> dp(static_cast<std::size_t>(k) + 1, 0);
    dp[0] = 1;  // c(0, 0) = 1
    for (std::int64_t i = 1; i <= n; ++i) {
        const std::int64_t top = std::min(i, k);
        for (std::int64_t j = top; j >= 1; --j) {
            const auto u = static_cast<std::size_t>(j);
            std::int64_t term = 0;
            if (mul_ov(i - 1, dp[u], term) || add_ov(term, dp[u - 1], dp[u])) {
                return make_error<std::int64_t>(MathError::overflow);
            }
        }
        dp[0] = 0;  // c(i, 0) = 0 for i >= 1
    }
    return dp[static_cast<std::size_t>(k)];
}

auto bernoulli(std::int64_t n) -> Result<Rational> {
    if (n < 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    // Akiyama-Tanigawa: seed row a[m] = 1/(m+1), then repeatedly apply
    // a[m] <- (m+1) * (a[m] - a[m+1]); the final a[0] is B_n. The row is shrunk one
    // element per pass, and each pass reads a[m], a[m+1] before overwriting a[m], so an
    // ascending in-place update is safe.
    const auto size = static_cast<std::size_t>(n) + 1;
    std::vector<Rational> a(size);
    for (std::size_t m = 0; m < size; ++m) {
        auto seed = Rational::make(1, static_cast<std::int64_t>(m) + 1);
        if (!seed) {
            return make_error<Rational>(seed.error());
        }
        a[m] = *seed;
    }
    for (std::int64_t j = 1; j <= n; ++j) {
        for (std::int64_t m = 0; m <= n - j; ++m) {
            const auto u = static_cast<std::size_t>(m);
            auto diff = a[u].subtract(a[u + 1]);
            if (!diff) {
                return make_error<Rational>(diff.error());
            }
            auto scaled = Rational::from_int(m + 1).multiply(*diff);
            if (!scaled) {
                return make_error<Rational>(scaled.error());
            }
            a[u] = *scaled;
        }
    }
    // The recurrence yields B_1 = +1/2; flip it to adopt the B_1 = -1/2 convention.
    // Every other Bernoulli number is identical under both conventions.
    if (n == 1) {
        return a[0].negate();
    }
    return a[0];
}

}  // namespace nimblecas
