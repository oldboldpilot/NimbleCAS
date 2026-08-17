# `nimblecas.modgcd` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/modgcd/modgcd.cppm`

Brown's modular GCD algorithm for univariate polynomials over $\mathbb{Z}[x]$ with
exact `int64` coefficients (ROADMAP §6.2). The algorithm reduces polynomial GCD
computation over $\mathbb{Z}[x]$ to independent modular images in $\mathbb{Z}_p[x]$
for single-word primes $p \in (2^{30}, 2^{31})$, lifts the reconstructed coefficients
via symmetric Chinese Remainder Theorem (CRT), and **strictly verifies the candidate
by exact trial division** before returning.

## Honesty boundary

This module is **exact and complete**:

- **Verified exactness**: Every reconstructed polynomial candidate $G$ is trial-divided
  into both primitive inputs using [`Polynomial::divide_exact`](polynomial.md). A
  candidate is accepted if and only if division succeeds with zero remainder for both
  inputs. The result is proven to be the exact GCD — never a probabilistic or approximate
  answer (Code Policy Rule 32).
- **Iteration budget**: If the prime budget (default 256 primes) is exhausted without
  producing a trial-division-verified candidate, the function returns an honest
  `MathError::not_converged`. It **never** returns an unverified candidate.
- **Dynamic range**: If any lifted coefficient or the content-scaled product exceeds
  the `int64` range, the operation returns `MathError::overflow`.
- **Pure mathematics**: `nimblecas.modgcd` contains no distributed transport or taskdag
  dependencies.

```cpp
import nimblecas.modgcd;
```

Depends on [`core`](core.md), [`polynomial`](polynomial.md), [`bigint`](bigint.md),
and [`numbertheory`](numbertheory.md).

## Data structures

### `ZpImage`

```cpp
struct ZpImage {
    std::uint64_t p{0};
    std::vector<std::uint64_t> coeffs{};
};
```

A $\gamma$-scaled monic polynomial GCD image in $\mathbb{Z}_p[x]$ where $p$ is a prime
in $(2^{30}, 2^{31})$ and `coeffs` are canonical residues in $[0, p)$, stored in
ascending-degree order trimmed of trailing zeros.

### `MergeOutcome`

```cpp
struct NeedMorePrimes {};
struct CoprimeProven {};
struct Candidate { Polynomial polynomial; };

using MergeOutcome = std::variant<NeedMorePrimes, CoprimeProven, Candidate>;
```

Represents the outcome of a symmetric CRT lift attempt over accumulated modular images:
- `NeedMorePrimes`: The product of accepted moduli has not yet exceeded $2 \cdot B_{\text{LM}}$,
  or more primes are required.
- `CoprimeProven`: An image with degree 0 was encountered, mathematically proving that
  the primitive parts are coprime ($\gcd = 1$).
- `Candidate`: Sufficient modular images have been lifted to form an integer polynomial
  candidate awaiting trial-division verification.

### `ImageRequest`

```cpp
struct ImageRequest {
    std::uint64_t p{0};
    std::int64_t gamma{0};
    Polynomial a{};
    Polynomial b{};
};
```

Wire representation for dispatching a per-prime GCD image calculation to a worker.

## Exported API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `landau_mignotte_bound` | `auto landau_mignotte_bound(const Polynomial& A, const Polynomial& B) -> Result<BigInt>` | Computes the square-free Landau–Mignotte coefficient bound $B_{\text{LM}} = \gamma \cdot 2^{\min(d_A, d_B)} \cdot \min((d_A+1)\|A\|_\infty, (d_B+1)\|B\|_\infty)$ as a `BigInt`. |
| `gcd_image_mod_p` | `auto gcd_image_mod_p(const Polynomial& A, const Polynomial& B, std::uint64_t p, std::int64_t gamma) -> Result<ZpImage>` | Reduces $A$ and $B$ modulo $p$, computes monic Euclidean GCD in $\mathbb{Z}_p[x]$, and scales by $\gamma \bmod p$. |
| `merge_images` | `auto merge_images(std::span<const ZpImage> images, std::int64_t gamma, const BigInt& b_lm) -> Result<MergeOutcome>` | Degree-filters images (unlucky-prime discard/reset), accumulates moduli until $\prod p_i > 2 \cdot B_{\text{LM}}$, performs coefficient-wise symmetric CRT, normalises to positive leading coefficient, and returns a `MergeOutcome`. |
| `modular_gcd` | `auto modular_gcd(const Polynomial& a, const Polynomial& b, std::size_t max_primes = 256) -> Result<Polynomial>` | In-process reference driver for Brown's modular GCD: content split, prime schedule iteration, symmetric CRT lift, and trial-division verification. |
| `encode_polynomial` | `auto encode_polynomial(const Polynomial& poly, std::vector<std::byte>& out) -> Result<void>` | Serializes an inner polynomial block ($n \times \text{u64}$ little-endian) into `out`. |
| `decode_polynomial` | `auto decode_polynomial(std::span<const std::byte> bytes, std::size_t& offset) -> Result<Polynomial>` | Deserializes an inner polynomial block starting at `offset`. |
| `encode_image_request` | `auto encode_image_request(const ImageRequest& req) -> Result<std::vector<std::byte>>` | Encodes an `ImageRequest` payload (magic `NCGQ`, version 1, little-endian). |
| `decode_image_request` | `auto decode_image_request(std::span<const std::byte> bytes) -> Result<ImageRequest>` | Decodes and validates an `ImageRequest` payload. |
| `encode_image_result` | `auto encode_image_result(const ZpImage& res) -> Result<std::vector<std::byte>>` | Encodes an `ImageResult` payload (magic `NCGI`, version 1, little-endian). |
| `decode_image_result` | `auto decode_image_result(std::span<const std::byte> bytes) -> Result<ZpImage>` | Decodes and validates an `ImageResult` payload. |

## Error model

| Condition | Error |
| :--- | :--- |
| Zero / degenerate inputs to `landau_mignotte_bound` or `gcd_image_mod_p` | `MathError::domain_error` |
| Leading coefficient divisible by prime modulus ($lc \equiv 0 \pmod p$) | `MathError::domain_error` |
| Lifted leading coefficient mismatch ($\text{lc}(G) \neq \gamma$) | `MathError::domain_error` |
| Malformed payload bytes (bad magic, version $\neq 1$, truncated or trailing bytes, untrimmed top coefficient) | `MathError::syntax_error` or `MathError::domain_error` |
| Prime budget exhausted without producing a trial-division-verified candidate | `MathError::not_converged` |
| Integer overflow during coefficient narrowing to `int64` or content scaling | `MathError::overflow` |

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.modgcd;

using namespace nimblecas;

// Polynomials: a = (x + 1)(x - 2) = x^2 - x - 2
//              b = (x + 1)(x + 3) = x^2 + 4x + 3
Polynomial a({-2, -1, 1});
Polynomial b({3, 4, 1});

// Exact modular GCD
auto g = modular_gcd(a, b).value();
// g == x + 1, coefficients: {1, 1}

// Content handling:
// a = 6(x^2 - 1) = 6x^2 - 6
// b = 4(x - 1)^2 = 4x^2 - 8x + 4
Polynomial c_a({-6, 0, 6});
Polynomial c_b({4, -8, 4});

auto g_content = modular_gcd(c_a, c_b).value();
// g_content == 2x - 2, coefficients: {-2, 2}

// Coprime polynomials:
Polynomial p1({1, 0, 1});   // x^2 + 1
Polynomial p2({-2, 0, 1});  // x^2 - 2

auto g_coprime = modular_gcd(p1, p2).value();
// g_coprime == 1, coefficients: {1}
```

## See also

- [`nimblecas.polynomial`](polynomial.md) — dense univariate integer polynomial arithmetic.
- [`nimblecas.numbertheory`](numbertheory.md) — Chinese Remainder Theorem and deterministic prime generation.
- [`nimblecas.bigint`](bigint.md) — arbitrary-precision arithmetic backing the symmetric CRT lift.
- [Documentation hub](../Index.md)
