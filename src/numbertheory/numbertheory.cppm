// NimbleCAS number theory & cryptographic primitives (on arbitrary-precision BigInt).
// @author Olumuyiwa Oluwasanmi
//
// The direct payoff of the BigInt foundation: exact, unbounded modular arithmetic and the
// building blocks of public-key cryptography. Everything here is EXACT arbitrary-precision
// integer arithmetic — there is no floating point and no silent overflow anywhere; every
// operation ultimately reduces to BigInt add/subtract/multiply (infallible) and
// divmod/modpow (railway-guarded). Failure is reported on the railway (Result<T> /
// MathError), never by throwing (Rule 32): a bad argument (non-positive modulus, a
// non-invertible element, mismatched or non-coprime CRT inputs, an even Jacobi modulus)
// surfaces as MathError::domain_error.
//
// HONESTY on primality: is_probable_prime uses Miller-Rabin with the fixed base set
// {2,3,5,7,11,13,17,19,23,29,31,37}. That base set is a *proven* deterministic primality
// test for every n < 3.317e24 (3'317'044'064'679'887'385'961'981) — below that bound the
// answer is exact. Above it the same bases plus a fixed number of extra random-base rounds
// make the test probabilistic: a composite survives r independent rounds with probability
// at most 4^-r, so the reported "prime" can (astronomically rarely) be a false positive,
// while "composite" is always certain.
//
// HONESTY on RSA: the rsa_* helpers are an EDUCATIONAL demonstration of the textbook RSA
// identity (m^(e·d) ≡ m (mod n)). They perform NO message padding (no OAEP / PKCS#1) and
// offer NO side-channel / constant-time guarantees. They are for teaching the mathematics
// only — do NOT use them to protect real data.

export module nimblecas.numbertheory;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.rng;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Extended Euclidean algorithm.
// ---------------------------------------------------------------------------
// The Bezout triple for a pair of integers: g == gcd(|a|, |b|) (normalised to g >= 0) and
// coefficients x, y with the exact identity a*x + b*y == g. Handles negative and zero
// inputs (extended_gcd(0, 0) yields g == 0, x == 1, y == 0).
struct ExtGcd {
    BigInt g;
    BigInt x;
    BigInt y;
};

// Iterative extended Euclid. Returns the Bezout triple {g, x, y} with a*x + b*y == g and
// g >= 0. Never fails: it is total over all integer pairs.
[[nodiscard]] auto extended_gcd(const BigInt& a, const BigInt& b) -> ExtGcd;

// ---------------------------------------------------------------------------
// Modular inverse.
// ---------------------------------------------------------------------------
// The multiplicative inverse of a modulo m, normalised to the canonical range [0, m):
// the unique t in [0, m) with (a * t) mod m == 1. Returns MathError::domain_error when
// m <= 0 (a modulus must be positive) or when gcd(a, m) != 1 (a is not invertible mod m).
[[nodiscard]] auto mod_inverse(const BigInt& a, const BigInt& m) -> Result<BigInt>;

// ---------------------------------------------------------------------------
// Primality (Miller-Rabin).
// ---------------------------------------------------------------------------
// Miller-Rabin primality test. Handles the small cases directly (n < 2 -> false,
// n in {2, 3} -> true, any other even n -> false), then writes n-1 = 2^s * d with d odd
// and screens n against the deterministic base set {2,3,5,7,11,13,17,19,23,29,31,37}.
//
// EXACTNESS: for every n < 3.317e24 that base set is a proven-correct primality test, so
// the result is DETERMINISTIC (exact). For n at or above that bound the routine ALSO runs
// a fixed number of random-base rounds drawn from Rng::seeded(seed); the answer is then
// probabilistic, with a composite escaping detection with probability at most 4^-rounds.
// "composite" is always certain; only "prime" carries that vanishing error above the bound.
[[nodiscard]] auto is_probable_prime(const BigInt& n, std::uint64_t seed) -> Result<bool>;

// The smallest (probable) prime strictly greater than n, found by scanning candidates with
// is_probable_prime (stepping by 2 once past the first odd candidate). `seed` seeds the
// Miller-Rabin witnesses used above the deterministic bound. n < 2 yields 2.
[[nodiscard]] auto next_prime(const BigInt& n, std::uint64_t seed) -> Result<BigInt>;

// ---------------------------------------------------------------------------
// Chinese Remainder Theorem.
// ---------------------------------------------------------------------------
// The unique x in [0, prod(moduli)) with x ≡ residues[i] (mod moduli[i]) for every i.
// Requires equal, non-empty lengths, every modulus >= 1, and the moduli PAIRWISE COPRIME;
// any violation yields MathError::domain_error.
[[nodiscard]] auto crt(const std::vector<BigInt>& residues, const std::vector<BigInt>& moduli)
    -> Result<BigInt>;

// ---------------------------------------------------------------------------
// Jacobi symbol.
// ---------------------------------------------------------------------------
// The Jacobi symbol (a / n) in {-1, 0, 1}, a generalisation of the Legendre symbol to any
// odd n > 1 (and (a/1) == 1). Requires n odd and positive, else MathError::domain_error.
// When n is an odd prime this equals the Legendre symbol; (a/n) == 0 iff gcd(a, n) > 1.
[[nodiscard]] auto jacobi_symbol(const BigInt& a, const BigInt& n) -> Result<int>;

// ---------------------------------------------------------------------------
// RSA — EDUCATIONAL demonstration only (see the module honesty note above).
// ---------------------------------------------------------------------------
// A textbook RSA key: modulus n = p*q, public exponent e, private exponent d with
// e*d ≡ 1 (mod lcm-substitute (p-1)(q-1)).
struct RsaKey {
    BigInt n;  // public modulus (product of two primes)
    BigInt e;  // public exponent
    BigInt d;  // private exponent
};

// Generate a demonstration RSA key of roughly `bits` modulus size from two random primes
// (each about bits/2 bits) drawn deterministically from `seed`. Returns domain_error if
// bits is too small to admit two distinct multi-bit primes (bits < 16). NOT secure key
// generation — no strong-prime tests, no padding, no side-channel hardening.
[[nodiscard]] auto rsa_generate(std::uint64_t bits, std::uint64_t seed) -> Result<RsaKey>;

// Textbook RSA "encrypt": message^e mod n. `message` must satisfy 0 <= message < n to be
// recoverable (this is a raw modular exponentiation with NO padding). Propagates the
// modpow railway (domain_error if n <= 0).
[[nodiscard]] auto rsa_encrypt(const BigInt& message, const BigInt& e, const BigInt& n)
    -> Result<BigInt>;

// Textbook RSA "decrypt": ciphertext^d mod n, the inverse of rsa_encrypt for a well-formed
// key and 0 <= message < n. No padding removal (there is none). Propagates modpow.
[[nodiscard]] auto rsa_decrypt(const BigInt& ciphertext, const BigInt& d, const BigInt& n)
    -> Result<BigInt>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Small BigInt constants, materialised once per call site where needed.
[[nodiscard]] auto bi_u64(std::uint64_t v) -> BigInt { return BigInt::from_u64(v); }

// Parity test. BigInt exposes no even/odd accessor, so parity is the remainder of a
// truncated division by 2 (the divisor is non-zero, so divmod always yields a value).
[[nodiscard]] auto is_even(const BigInt& x) -> bool {
    auto dm = x.divmod(bi_u64(2));
    return dm.has_value() && dm->second.is_zero();
}

// Exact halving of an even value (its truncated quotient by 2). The divisor is non-zero.
[[nodiscard]] auto halve(const BigInt& x) -> BigInt {
    auto dm = x.divmod(bi_u64(2));
    return dm.has_value() ? dm->first : BigInt{};
}

// Reduce x into the canonical residue [0, m) for m > 0. The BigInt remainder takes the
// dividend's sign (truncation toward zero), so a negative remainder is folded up by one m.
// PRECONDITION: m > 0 (all call sites guarantee it).
[[nodiscard]] auto mod_floor(const BigInt& x, const BigInt& m) -> BigInt {
    auto dm = x.divmod(m);
    if (!dm.has_value()) {
        return BigInt{};  // unreachable while m > 0; keeps the helper total
    }
    BigInt r = std::move(dm->second);
    if (r.is_negative()) {
        r = r.add(m);
    }
    return r;
}

// 2^64 as a BigInt: the radix used to pack successive 64-bit RNG draws into a big integer.
[[nodiscard]] auto two_pow_64() -> BigInt { return bi_u64(2).pow(64); }

// Assemble a non-negative BigInt from `words` fresh 64-bit draws (big-endian packing).
// The magnitude is uniform over [0, 2^(64*words)); callers reduce it into the range they
// need. Consumes exactly `words` draws from `rng`.
[[nodiscard]] auto random_bigint(Rng& rng, std::size_t words) -> BigInt {
    const BigInt radix = two_pow_64();
    BigInt acc{};
    for (std::size_t i = 0; i < words; ++i) {
        acc = acc.multiply(radix).add(bi_u64(rng.next_u64()));
    }
    return acc;
}

// The deterministic Miller-Rabin base set: a proven-correct witness list for n < 3.317e24.
constexpr std::array<std::uint64_t, 12> kDeterministicBases = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

// The proven deterministic bound for kDeterministicBases: n strictly below this value is
// prime iff it passes every base in the set (3'317'044'064'679'887'385'961'981).
[[nodiscard]] auto deterministic_bound() -> BigInt {
    return BigInt::from_string("3317044064679887385961981").value();
}

// Number of extra random-base rounds run above the deterministic bound. Composite escape
// probability is at most 4^-40, i.e. below 2^-80.
constexpr std::uint64_t kRandomRounds = 40;

// One Miller-Rabin round for a single base `a` in [2, n-2], given the decomposition
// n - 1 == 2^s * d (d odd) and n_minus_1 == n - 1. Returns true when `a` is not a witness
// to compositeness (n survives the round), false when `a` proves n composite. Any BigInt
// railway failure (never expected here, n > 0) is propagated.
[[nodiscard]] auto miller_rabin_round(const BigInt& n, const BigInt& n_minus_1,
                                      const BigInt& d, std::uint64_t s, const BigInt& a)
    -> Result<bool> {
    auto xr = a.modpow(d, n);
    if (!xr) {
        return make_error<bool>(xr.error());
    }
    BigInt x = std::move(*xr);
    const BigInt one = bi_u64(1);
    if (x == one || x == n_minus_1) {
        return true;
    }
    for (std::uint64_t i = 1; i < s; ++i) {
        auto sq = x.multiply(x).mod(n);
        if (!sq) {
            return make_error<bool>(sq.error());
        }
        x = std::move(*sq);
        if (x == n_minus_1) {
            return true;
        }
    }
    return false;  // `a` is a witness: n is composite
}

}  // namespace

// --- Extended Euclid --------------------------------------------------------

auto extended_gcd(const BigInt& a, const BigInt& b) -> ExtGcd {
    // Standard iterative form maintaining the invariants
    //   a*old_s + b*old_t == old_r  and  a*s + b*t == r
    // throughout. Truncated division keeps |r| strictly decreasing, so the loop always
    // reaches r == 0 (even for negative inputs), leaving old_r == ±gcd.
    BigInt old_r = a;
    BigInt r = b;
    BigInt old_s = bi_u64(1);
    BigInt s{};  // 0
    BigInt old_t{};  // 0
    BigInt t = bi_u64(1);

    while (!r.is_zero()) {
        // r != 0 here, so divmod yields a value; q is the truncated quotient old_r / r.
        auto dm = old_r.divmod(r);
        const BigInt q = dm.has_value() ? dm->first : BigInt{};

        BigInt next_r = old_r.subtract(q.multiply(r));
        old_r = std::move(r);
        r = std::move(next_r);

        BigInt next_s = old_s.subtract(q.multiply(s));
        old_s = std::move(s);
        s = std::move(next_s);

        BigInt next_t = old_t.subtract(q.multiply(t));
        old_t = std::move(t);
        t = std::move(next_t);
    }

    // Normalise g >= 0; negating the whole triple preserves a*x + b*y == g.
    if (old_r.is_negative()) {
        old_r = old_r.negate();
        old_s = old_s.negate();
        old_t = old_t.negate();
    }
    return ExtGcd{std::move(old_r), std::move(old_s), std::move(old_t)};
}

// --- Modular inverse --------------------------------------------------------

auto mod_inverse(const BigInt& a, const BigInt& m) -> Result<BigInt> {
    if (m.sign() <= 0) {
        return make_error<BigInt>(MathError::domain_error);  // modulus must be positive
    }
    const ExtGcd eg = extended_gcd(a, m);
    if (eg.g != bi_u64(1)) {
        return make_error<BigInt>(MathError::domain_error);  // a not invertible mod m
    }
    // a * eg.x ≡ 1 (mod m); fold eg.x into the canonical range [0, m).
    return mod_floor(eg.x, m);
}

// --- Primality --------------------------------------------------------------

auto is_probable_prime(const BigInt& n, std::uint64_t seed) -> Result<bool> {
    const BigInt two = bi_u64(2);
    const BigInt three = bi_u64(3);

    if (n < two) {
        return false;  // 0, 1, and all negatives are non-prime
    }
    if (n == two || n == three) {
        return true;
    }
    if (is_even(n)) {
        return false;  // any even n > 2 is composite
    }

    // Decompose n - 1 = 2^s * d with d odd.
    const BigInt n_minus_1 = n.subtract(bi_u64(1));
    BigInt d = n_minus_1;
    std::uint64_t s = 0;
    while (is_even(d)) {
        d = halve(d);
        ++s;
    }

    // Deterministic base screen. Bases >= n are skipped (only possible for tiny n, which the
    // small-case handling above has already resolved for the true primes).
    for (const std::uint64_t base : kDeterministicBases) {
        const BigInt a = bi_u64(base);
        if (!(a < n)) {
            continue;
        }
        auto passed = miller_rabin_round(n, n_minus_1, d, s, a);
        if (!passed) {
            return make_error<bool>(passed.error());
        }
        if (!*passed) {
            return false;  // certain composite
        }
    }

    // Below the proven bound the deterministic screen is exact — accept now.
    if (n < deterministic_bound()) {
        return true;
    }

    // Above the bound, reinforce with random-base rounds. Witnesses are drawn uniformly-ish
    // from [2, n-2]: build a big random value and reduce it modulo (n-3), then add 2.
    Rng rng = Rng::seeded(seed);
    const BigInt range = n.subtract(three);  // n >= 5 and odd here, so range >= 2
    const std::size_t words = n.to_string().size() / 19 + 2;  // enough entropy to cover n
    for (std::uint64_t round = 0; round < kRandomRounds; ++round) {
        const BigInt r = random_bigint(rng, words);
        const BigInt a = mod_floor(r, range).add(two);  // in [2, n-2]
        auto passed = miller_rabin_round(n, n_minus_1, d, s, a);
        if (!passed) {
            return make_error<bool>(passed.error());
        }
        if (!*passed) {
            return false;
        }
    }
    return true;  // probable prime (error <= 4^-kRandomRounds)
}

auto next_prime(const BigInt& n, std::uint64_t seed) -> Result<BigInt> {
    const BigInt two = bi_u64(2);
    if (n < two) {
        return two;  // the smallest prime
    }
    // Start just above n and settle on an odd candidate (2 is already excluded since n >= 2).
    BigInt cand = n.add(bi_u64(1));
    if (is_even(cand)) {
        cand = cand.add(bi_u64(1));  // step to the next odd
    }
    while (true) {
        auto prime = is_probable_prime(cand, seed);
        if (!prime) {
            return make_error<BigInt>(prime.error());
        }
        if (*prime) {
            return cand;
        }
        cand = cand.add(two);  // only odd candidates need testing
    }
}

// --- Chinese Remainder Theorem ----------------------------------------------

auto crt(const std::vector<BigInt>& residues, const std::vector<BigInt>& moduli)
    -> Result<BigInt> {
    if (residues.empty() || residues.size() != moduli.size()) {
        return make_error<BigInt>(MathError::domain_error);  // equal, non-empty lengths
    }
    const BigInt one = bi_u64(1);
    // Every modulus must be >= 1.
    for (const BigInt& m : moduli) {
        if (m < one) {
            return make_error<BigInt>(MathError::domain_error);
        }
    }
    // Moduli must be pairwise coprime.
    for (std::size_t i = 0; i < moduli.size(); ++i) {
        for (std::size_t j = i + 1; j < moduli.size(); ++j) {
            if (BigInt::gcd(moduli[i], moduli[j]) != one) {
                return make_error<BigInt>(MathError::domain_error);
            }
        }
    }

    // Fold the congruences together. Invariant: x is the unique solution in [0, M) of the
    // congruences processed so far, where M is their product.
    BigInt x = mod_floor(residues[0], moduli[0]);
    BigInt M = moduli[0];
    for (std::size_t i = 1; i < moduli.size(); ++i) {
        const BigInt& mi = moduli[i];
        // Solve x' ≡ x (mod M) and x' ≡ residues[i] (mod mi) as x' = x + M * k, where
        // k = (residues[i] - x) * (M^{-1} mod mi)  (mod mi). Coprimality guarantees M^{-1}.
        const BigInt M_red = mod_floor(M, mi);
        auto inv = mod_inverse(M_red, mi);
        if (!inv) {
            return make_error<BigInt>(inv.error());  // unreachable given pairwise coprimality
        }
        const BigInt diff = mod_floor(residues[i].subtract(x), mi);
        const BigInt k = mod_floor(diff.multiply(*inv), mi);
        x = x.add(M.multiply(k));  // stays in [0, M*mi) by construction
        M = M.multiply(mi);
    }
    return x;
}

// --- Jacobi symbol ----------------------------------------------------------

auto jacobi_symbol(const BigInt& a, const BigInt& n) -> Result<int> {
    if (n.sign() <= 0 || is_even(n)) {
        return make_error<int>(MathError::domain_error);  // n must be odd and positive
    }
    const BigInt one = bi_u64(1);
    const BigInt three = bi_u64(3);
    const BigInt four = bi_u64(4);
    const BigInt five = bi_u64(5);
    const BigInt eight = bi_u64(8);

    BigInt num = mod_floor(a, n);  // work with a reduced into [0, n)
    BigInt den = n;
    int result = 1;

    while (!num.is_zero()) {
        // Pull out factors of 2: each flips the sign iff den ≡ 3 or 5 (mod 8) (the
        // supplement (2/den) = (-1)^((den^2-1)/8)).
        while (is_even(num)) {
            num = halve(num);
            const BigInt r8 = mod_floor(den, eight);
            if (r8 == three || r8 == five) {
                result = -result;
            }
        }
        // Quadratic reciprocity: swapping introduces a sign flip iff both ≡ 3 (mod 4).
        std::swap(num, den);
        if (mod_floor(num, four) == three && mod_floor(den, four) == three) {
            result = -result;
        }
        num = mod_floor(num, den);
    }
    return den == one ? result : 0;  // (a/n) == 0 exactly when gcd(a, n) > 1
}

// --- RSA (educational) ------------------------------------------------------

namespace {

// A random prime of approximately `bits` bits, drawn from `rng`, using `seed` for the
// Miller-Rabin witnesses inside next_prime. Builds a random value in [2^(bits-1), 2^bits)
// and returns the next probable prime at or above it. PRECONDITION: bits >= 2.
[[nodiscard]] auto random_prime(Rng& rng, std::uint64_t bits, std::uint64_t seed)
    -> Result<BigInt> {
    const BigInt high = bi_u64(2).pow(bits - 1);          // 2^(bits-1)
    const std::size_t words = static_cast<std::size_t>(bits / 64) + 2;
    const BigInt r = random_bigint(rng, words);
    const BigInt cand = high.add(mod_floor(r, high));      // in [2^(bits-1), 2^bits)
    // next_prime returns the smallest prime strictly greater than its argument, so pass
    // cand - 1 to allow cand itself to be selected when it is already prime.
    return next_prime(cand.subtract(bi_u64(1)), seed);
}

}  // namespace

auto rsa_generate(std::uint64_t bits, std::uint64_t seed) -> Result<RsaKey> {
    if (bits < 16) {
        return make_error<RsaKey>(MathError::domain_error);  // too small for two distinct primes
    }
    Rng rng = Rng::seeded(seed);
    const std::uint64_t pbits = bits / 2;
    const std::uint64_t qbits = bits - pbits;

    Rng prng = rng.split(1);
    Rng qrng = rng.split(2);
    auto p_res = random_prime(prng, pbits, seed ^ 0x50ULL);
    if (!p_res) {
        return make_error<RsaKey>(p_res.error());
    }
    const BigInt p = *p_res;

    // Draw q, resampling in the (tiny-probability) event it collides with p.
    BigInt q;
    while (true) {
        auto q_res = random_prime(qrng, qbits, seed ^ 0x51ULL);
        if (!q_res) {
            return make_error<RsaKey>(q_res.error());
        }
        if (*q_res != p) {
            q = *q_res;
            break;
        }
    }

    const BigInt one = bi_u64(1);
    const BigInt n = p.multiply(q);
    const BigInt phi = p.subtract(one).multiply(q.subtract(one));  // (p-1)(q-1)

    // Choose a public exponent coprime to phi, starting at the customary 65537 and, if
    // that shares a factor, walking up through the primes until one is coprime.
    BigInt e = bi_u64(65537);
    while (BigInt::gcd(e, phi) != one) {
        auto ne = next_prime(e, seed);
        if (!ne) {
            return make_error<RsaKey>(ne.error());
        }
        e = *ne;
    }
    auto d = mod_inverse(e, phi);
    if (!d) {
        return make_error<RsaKey>(d.error());  // unreachable: e is coprime to phi
    }
    return RsaKey{n, e, *d};
}

auto rsa_encrypt(const BigInt& message, const BigInt& e, const BigInt& n) -> Result<BigInt> {
    return message.modpow(e, n);
}

auto rsa_decrypt(const BigInt& ciphertext, const BigInt& d, const BigInt& n) -> Result<BigInt> {
    return ciphertext.modpow(d, n);
}

}  // namespace nimblecas
