# `nimblecas.combinatorics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/combinatorics/combinatorics.cppm`

Exact, overflow-checked counting functions over `int64` — factorial, binomial,
permutations, Catalan, Fibonacci, and both kinds of Stirling number — plus
**Bernoulli numbers** as exact reduced [`Rational`](ratpoly.md)s (ROADMAP
§7.18). Every function is a free function returning `Result`; nothing throws.

```cpp
import nimblecas.combinatorics;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): when an `int64` result would exceed the
representable range the operation returns `MathError::overflow` rather than
silently wrapping, and a mathematically undefined argument (e.g. a negative
factorial) returns `MathError::domain_error`. The integer routines are written
to keep intermediate magnitudes as small as the exact answer allows — `binomial`
and `catalan` interleave a division after every multiplication, so the running
value is always an exact integer that never exceeds the final answer, both
avoiding spurious overflow and keeping every step within `int64` whenever the
final answer is.

## Integer sequences

All return `Result<std::int64_t>`, are exact, and are overflow-checked.

| Function | Signature | Meaning & domain |
| :--- | :--- | :--- |
| `factorial` | `auto factorial(std::int64_t n) -> Result<std::int64_t>` | `n!` for `n >= 0`. `domain_error` for `n < 0`; `overflow` for `n > 20` (21! exceeds `INT64_MAX`). |
| `binomial` | `auto binomial(std::int64_t n, std::int64_t k) -> Result<std::int64_t>` | `C(n, k) = n! / (k!(n-k)!)`. Returns `0` when `k < 0` or `k > n` (the conventional empty count); `domain_error` for `n < 0`; `overflow` past `INT64_MAX`. Multiplicative formula, divide-as-you-go. |
| `permutations` | `auto permutations(std::int64_t n, std::int64_t k) -> Result<std::int64_t>` | `P(n, k) = n!/(n-k)!`, the falling factorial `n(n-1)…(n-k+1)`. Returns `0` when `k < 0` or `k > n`; `domain_error` for `n < 0`; `overflow` past `INT64_MAX`. |
| `catalan` | `auto catalan(std::int64_t n) -> Result<std::int64_t>` | The `n`-th Catalan number `C_n = C(2n, n)/(n+1)`. `domain_error` for `n < 0`; `overflow` once the value exceeds `INT64_MAX` (`n >= 36`). |
| `fibonacci` | `auto fibonacci(std::int64_t n) -> Result<std::int64_t>` | The `n`-th Fibonacci number (`F_0 = 0`, `F_1 = 1`), iterative. `domain_error` for `n < 0`; `overflow` once the value exceeds `INT64_MAX` (`n >= 93`). |
| `stirling_second` | `auto stirling_second(std::int64_t n, std::int64_t k) -> Result<std::int64_t>` | Stirling number of the second kind `S(n, k)`: partitions of `n` labelled elements into `k` non-empty unlabelled subsets. Returns `0` when `k > n`; `domain_error` for `n < 0` or `k < 0`; `overflow` past `INT64_MAX`. |
| `stirling_first` | `auto stirling_first(std::int64_t n, std::int64_t k) -> Result<std::int64_t>` | Unsigned Stirling number of the first kind `c(n, k)`: permutations of `n` elements with exactly `k` disjoint cycles. Returns `0` when `k > n`; `domain_error` for `n < 0` or `k < 0`; `overflow` past `INT64_MAX`. |

## Bernoulli numbers — `bernoulli`

```cpp
[[nodiscard]] auto bernoulli(std::int64_t n) -> Result<Rational>;
```

The `n`-th Bernoulli number as an exact reduced fraction, computed with the
**Akiyama–Tanigawa** algorithm over exact `Rational`: seed a row
`a[m] = 1/(m+1)`, then repeatedly apply `a[m] <- (m+1)·(a[m] - a[m+1])`; the
final `a[0]` is `B_n`.

That recurrence natively yields the "second" Bernoulli numbers with
`B_1 = +1/2`; every other `B_n` is convention-independent. This module adopts
the more common **first convention `B_1 = -1/2`** by flipping the sign of that
single value. So `B_0 = 1`, `B_1 = -1/2`, `B_2 = 1/6`, and `B_{2k+1} = 0` for
`k >= 1`. Returns `domain_error` for `n < 0`; `overflow` if any intermediate
`Rational` exceeds `int64`.

## Error model

| Condition | Error |
| :--- | :--- |
| Negative `n` (`factorial`, `binomial`, `permutations`, `catalan`, `fibonacci`, `bernoulli`) | `MathError::domain_error` |
| Negative `n` or `k` (`stirling_second`, `stirling_first`) | `MathError::domain_error` |
| Exact result exceeds `INT64_MAX` (integer sequences) | `MathError::overflow` |
| An intermediate `Rational` exceeds `int64` (`bernoulli`) | `MathError::overflow` |

Out-of-range-but-defined counts are **not** errors: `binomial`, `permutations`,
`stirling_second`, and `stirling_first` return `0` for `k < 0` or `k > n` (the
conventional empty count).

## Worked values

```cpp
import nimblecas.combinatorics;
import nimblecas.ratpoly;
using namespace nimblecas;

factorial(10).value();        // 3628800
factorial(20).value();        // 2432902008176640000  (largest factorial in int64)
factorial(21).error();        // MathError::overflow

binomial(52, 5).value();      // 2598960  (poker hands)
binomial(6, 3).value();       // 20
binomial(5, 7).value();       // 0        (k > n)

permutations(5, 3).value();   // 60
catalan(0..5);                // 1, 1, 2, 5, 14, 42
fibonacci(92).value();        // 7540113804746346429  (largest Fibonacci in int64)

stirling_second(4, 2).value();// 7
stirling_first(4, 2).value(); // 11

bernoulli(0).value();         // 1
bernoulli(1).value();         // -1/2   (first convention)
bernoulli(2).value();         // 1/6
bernoulli(3).value();         // 0      (odd B_{2k+1} = 0)
bernoulli(4).value();         // -1/30
bernoulli(6).value();         // 1/42
bernoulli(10).value();        // 5/66
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` type Bernoulli
  numbers are returned in.
- [`nimblecas.matrix`](matrix.md) and [`nimblecas.orthopoly`](orthopoly.md) —
  the sibling `ratpoly`-consuming numeric modules.
- [Documentation hub](../Index.md)
