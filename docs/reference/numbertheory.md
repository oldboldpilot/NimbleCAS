# `nimblecas.numbertheory` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/numbertheory/numbertheory.cppm`

Number theory and public-key cryptographic primitives built on the
arbitrary-precision [`BigInt`](bigint.md). This is the direct payoff of the
`BigInt` foundation: **exact, unbounded modular arithmetic** and the building
blocks of RSA. Everything here is exact arbitrary-precision integer
arithmetic — there is **no floating point and no silent overflow anywhere**.
Every operation reduces to `BigInt` add / subtract / multiply (infallible) and
divmod / modpow (railway-guarded), so failure is reported on the railway
(`Result<T>` / `MathError`), never by throwing (Rule 32). A bad argument — a
non-positive modulus, a non-invertible element, mismatched or non-coprime CRT
inputs, an even Jacobi modulus, a too-small RSA key size — surfaces as
`MathError::domain_error`.

Two honesty boundaries are load-bearing and must not be forgotten:

- **Primality is exact only below a proven bound.** `is_probable_prime` runs
  Miller–Rabin against the fixed base set `{2,3,5,7,11,13,17,19,23,29,31,37}`.
  That set is a *proven* deterministic primality test for every
  `n < 3.317e24` (exactly `3'317'044'064'679'887'385'961'981`) — below that
  bound the answer is **exact**. At or above it the routine adds 40 random-base
  rounds seeded from `Rng::seeded(seed)`; the answer is then **probabilistic**:
  a composite escapes detection with probability at most `4^-40 = 2^-80`.
  `"composite"` is always certain; only `"prime"` carries that vanishing error,
  and only above the bound.
- **The RSA helpers are an educational demonstration only.** `rsa_generate` /
  `rsa_encrypt` / `rsa_decrypt` exhibit the textbook identity
  `m^(e·d) ≡ m (mod n)`. They perform **no** message padding (no OAEP /
  PKCS#1) and offer **no** side-channel / constant-time guarantees. They teach
  the mathematics — do **not** use them to protect real data.

```cpp
import nimblecas.numbertheory;
```

Depends on [`core`](core.md) (the `Result` / `MathError` railway),
[`bigint`](bigint.md) (the exact integer type every argument and result is
built from), and `nimblecas.rng` (the seeded PRNG driving the random
Miller–Rabin witnesses and RSA prime search).

## Extended Euclid and modular inverse

The one **infallible** routine in the module: `extended_gcd` is total over all
integer pairs.

### `ExtGcd` — a Bezout triple

```cpp
struct ExtGcd {
    BigInt g;
    BigInt x;
    BigInt y;
};
```

`g == gcd(|a|, |b|)` normalised to `g >= 0`, with coefficients satisfying the
exact identity `a*x + b*y == g`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `extended_gcd` | `[[nodiscard]] auto extended_gcd(const BigInt& a, const BigInt& b) -> ExtGcd` | Iterative extended Euclid. Returns `{g, x, y}` with `a*x + b*y == g` and `g >= 0`. Handles negative and zero inputs; `extended_gcd(0, 0)` yields `g == 0, x == 1, y == 0`. **Never fails.** |
| `mod_inverse` | `[[nodiscard]] auto mod_inverse(const BigInt& a, const BigInt& m) -> Result<BigInt>` | The multiplicative inverse of `a` modulo `m`, normalised to the canonical range `[0, m)`: the unique `t` in `[0, m)` with `(a * t) mod m == 1`. `domain_error` when `m <= 0` or when `gcd(a, m) != 1` (i.e. `a` is not invertible mod `m`). |

## Primality

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_probable_prime` | `[[nodiscard]] auto is_probable_prime(const BigInt& n, std::uint64_t seed) -> Result<bool>` | Miller–Rabin. Small cases resolve directly (`n < 2 -> false`, `n in {2,3} -> true`, any other even `n -> false`). Then screens `n` against the deterministic base set. **Exact** (`Result` holds a certain answer) for every `n < 3.317e24`; at or above the bound it adds 40 random-base rounds from `Rng::seeded(seed)` and the `true` answer becomes probabilistic (composite escape probability `<= 4^-40`). `false` is always certain. |
| `next_prime` | `[[nodiscard]] auto next_prime(const BigInt& n, std::uint64_t seed) -> Result<BigInt>` | The smallest (probable) prime strictly greater than `n`, found by scanning candidates with `is_probable_prime` (stepping by 2 past the first odd candidate). `seed` seeds the Miller–Rabin witnesses used above the deterministic bound. `n < 2` yields `2`. |

The `seed` parameter is only consulted above the deterministic bound; below it
the result is independent of `seed`.

## Chinese Remainder Theorem and Jacobi symbol

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `crt` | `[[nodiscard]] auto crt(const std::vector<BigInt>& residues, const std::vector<BigInt>& moduli) -> Result<BigInt>` | The unique `x` in `[0, prod(moduli))` with `x ≡ residues[i] (mod moduli[i])` for every `i`. Requires equal, non-empty lengths, every modulus `>= 1`, and the moduli **pairwise coprime**; any violation yields `domain_error`. |
| `jacobi_symbol` | `[[nodiscard]] auto jacobi_symbol(const BigInt& a, const BigInt& n) -> Result<int>` | The Jacobi symbol `(a / n)` in `{-1, 0, 1}`, a generalisation of the Legendre symbol to any odd `n > 1` (with `(a/1) == 1`). Requires `n` odd and positive, else `domain_error`. When `n` is an odd prime this equals the Legendre symbol; `(a/n) == 0` iff `gcd(a, n) > 1`. A negative `a` is reduced mod `n` first. |

## RSA — educational demonstration only

See the module honesty note above: no padding, no constant-time guarantees, not
secure key generation.

### `RsaKey` — a textbook RSA key

```cpp
struct RsaKey {
    BigInt n;  // public modulus (product of two primes)
    BigInt e;  // public exponent
    BigInt d;  // private exponent
};
```

`n = p*q`, public exponent `e`, private exponent `d` with `e*d ≡ 1` modulo
`(p-1)(q-1)`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `rsa_generate` | `[[nodiscard]] auto rsa_generate(std::uint64_t bits, std::uint64_t seed) -> Result<RsaKey>` | A demonstration key of roughly `bits` modulus size from two random primes (each about `bits/2` bits) drawn deterministically from `seed`. Public exponent starts at the customary `65537` and walks up through primes until coprime to `(p-1)(q-1)`. `domain_error` if `bits < 16` (too small for two distinct multi-bit primes). Not secure: no strong-prime tests, no padding, no side-channel hardening. |
| `rsa_encrypt` | `[[nodiscard]] auto rsa_encrypt(const BigInt& message, const BigInt& e, const BigInt& n) -> Result<BigInt>` | Textbook "encrypt": `message^e mod n`, a raw modular exponentiation with **no** padding. `message` must satisfy `0 <= message < n` to be recoverable. Propagates the `modpow` railway (`domain_error` if `n <= 0`). |
| `rsa_decrypt` | `[[nodiscard]] auto rsa_decrypt(const BigInt& ciphertext, const BigInt& d, const BigInt& n) -> Result<BigInt>` | Textbook "decrypt": `ciphertext^d mod n`, the inverse of `rsa_encrypt` for a well-formed key and `0 <= message < n`. No padding removal (there is none). Propagates `modpow`. |

## Error model

All fallible entry points report on the `Result` railway; none throw.

| Condition | Error |
| :--- | :--- |
| `mod_inverse` with `m <= 0` | `MathError::domain_error` |
| `mod_inverse` with `gcd(a, m) != 1` (non-invertible) | `MathError::domain_error` |
| `crt` with empty or mismatched-length inputs | `MathError::domain_error` |
| `crt` with any modulus `< 1` | `MathError::domain_error` |
| `crt` with non-pairwise-coprime moduli | `MathError::domain_error` |
| `jacobi_symbol` with `n` even or `n <= 0` | `MathError::domain_error` |
| `rsa_generate` with `bits < 16` | `MathError::domain_error` |
| `rsa_encrypt` / `rsa_decrypt` with modulus `n <= 0` | `MathError::domain_error` (propagated from `modpow`) |

`extended_gcd` is total and returns a plain `ExtGcd` (never fails).
`is_probable_prime` and `next_prime` only surface an error if an underlying
`BigInt` railway operation fails, which does not occur for valid inputs; note
that "not prime" is a `false` **value**, not an error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.numbertheory;
using namespace nimblecas;

auto bi = [](std::string_view s) { return BigInt::from_string(s).value(); };
constexpr std::uint64_t seed = 0x9E3779B97F4A7C15ULL;

// Extended Euclid: gcd(240, 46) == 2, with the exact Bezout identity.
const ExtGcd eg = extended_gcd(bi("240"), bi("46"));
eg.g.to_string();                                     // "2"
bi("240").multiply(eg.x).add(bi("46").multiply(eg.y)) == eg.g;  // true

// Modular inverse, normalised into [0, m).
mod_inverse(bi("3"), bi("11")).value();               // 4   (3*4 == 12 ≡ 1 mod 11)
mod_inverse(bi("17"), bi("3120")).value();            // 2753
mod_inverse(bi("4"), bi("6")).error();                // domain_error (gcd == 2)
mod_inverse(bi("3"), bi("0")).error();                // domain_error (modulus 0)

// Primality: deterministic below the bound, probabilistic above it.
is_probable_prime(bi("2305843009213693951"), seed).value();  // true  (2^61 - 1)
is_probable_prime(bi("561"), seed).value();           // false (Carmichael 3*11*17)
is_probable_prime(bi("-7"), seed).value();            // false
// 10^99 + 289, the smallest 100-digit prime — exercises the random-witness branch.
const std::string p100 = std::string("1") + std::string(96, '0') + "289";
is_probable_prime(bi(p100), seed).value();            // true

// next_prime.
next_prime(bi("90"), seed).value();                   // 97
next_prime(bi("0"), seed).value();                    // 2
next_prime(bi("-5"), seed).value();                   // 2

// Chinese Remainder Theorem: x ≡ 2 (3), 3 (5), 2 (7) -> 23.
auto x = crt({bi("2"), bi("3"), bi("2")}, {bi("3"), bi("5"), bi("7")}).value();  // 23
crt({bi("2"), bi("3")}, {bi("4"), bi("6")}).error();  // domain_error (non-coprime)
crt({}, {}).error();                                  // domain_error (empty)

// Jacobi symbol.
jacobi_symbol(bi("1001"), bi("9907")).value();        // -1
jacobi_symbol(bi("6"), bi("9")).value();              // 0  (gcd(6, 9) == 3)
jacobi_symbol(bi("-1"), bi("15")).value();            // -1 (a reduced mod n first)
jacobi_symbol(bi("3"), bi("4")).error();              // domain_error (even modulus)

// Educational RSA round-trip: generate -> encrypt -> decrypt.
auto key = rsa_generate(64, 0xC0FFEEULL).value();
const BigInt m = bi("123456789");                     // must satisfy 0 <= m < key.n
auto c    = rsa_encrypt(m, key.e, key.n).value();
auto back = rsa_decrypt(c, key.d, key.n).value();     // == m
rsa_generate(8, 1).error();                           // domain_error (bits < 16)
```

## See also

- [`nimblecas.bigint`](bigint.md) — the exact arbitrary-precision integer every
  operation here is built on.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway.
- [`nimblecas.bigrational`](bigrational.md) and
  [`nimblecas.bigcombinatorics`](bigcombinatorics.md) — sibling `BigInt`-consuming
  modules.
- [Documentation hub](../Index.md)
