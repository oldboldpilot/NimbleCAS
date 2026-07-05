# `nimblecas.bigcombinatorics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigcombinatorics/bigcombinatorics.cppm`

The **unbounded, overflow-free** counterpart of
[`nimblecas.combinatorics`](combinatorics.md). Every count is carried in a
`BigInt` (see `nimblecas.bigint`), which allocates as many limbs as the exact
answer needs, so **none of these routines can overflow**. Where the `int64`
module reports `MathError::overflow` the instant an exact answer would leave the
representable range — `factorial` past `20!`, `fibonacci` past `F(92)`,
`catalan` from `n = 36` — the functions here simply keep counting. The trade-off
is honest and deliberate: the `int64` module is faster (native machine words),
this one is unbounded (arbitrary precision). A negative argument or an
out-of-range `k` is still a `MathError::domain_error`, matching the `int64`
module function-for-function; there is simply **no overflow error to return**.

Scope is confined to the **integer-valued** sequences. Rational-valued
sequences such as the Bernoulli numbers are deliberately absent — they need a
`BigRational`-backed module, not this one; the `int64`
[`nimblecas.combinatorics`](combinatorics.md) still offers `bernoulli()` over
exact (bounded) `Rational`.

```cpp
import nimblecas.bigcombinatorics;
```

Depends on [`core`](core.md) and `nimblecas.bigint`.

## The exactness contract

Exactness is preserved exactly as in the `int64` module: the multiplicative
`binomial` / `catalan` / `multinomial` formulas interleave an **exact division
after each multiplication**, so the running value is always an integer that
never exceeds the final answer (never the wasteful
`factorial/factorial/factorial`). Because `BigInt::divmod` is the only division
available, each such divide is performed as a `divmod` and the remainder is
checked on the **railway** — a would-be inexact division surfaces as
`MathError::undefined_value` rather than silently truncating under `-DNDEBUG`.
Exactness holds by construction at every call site, so `undefined_value` is an
invariant-violation signal that a correct build never emits.

Every function is a free function in `namespace nimblecas` returning
`Result<BigInt>`; nothing throws.

## Sequences of a single index

| Function | Signature | Meaning & domain |
| :--- | :--- | :--- |
| `factorial` | `[[nodiscard]] auto factorial(std::int64_t n) -> Result<BigInt>` | `n!` as the iterative product `1·2·…·n` (`0! = 1`). `domain_error` for `n < 0`. No overflow ceiling: `25!` and `50!` are exact here, whereas `21!` already overflows `int64`. |
| `catalan` | `[[nodiscard]] auto catalan(std::int64_t n) -> Result<BigInt>` | The `n`-th Catalan number `C_n = C(2n, n)/(n+1)`, exact. `domain_error` for `n < 0` **or** when `2n`/`n+1` would overflow the `int64` argument. Keeps counting where the `int64` `catalan` overflows at `n = 36`. |
| `double_factorial` | `[[nodiscard]] auto double_factorial(std::int64_t n) -> Result<BigInt>` | `n!! = n(n-2)(n-4)…` down to `1` (odd) or `2` (even), with the empty-product convention `(-1)!! = 0!! = 1`. Requires `n >= -1`, else `domain_error`. |
| `fibonacci` | `[[nodiscard]] auto fibonacci(std::int64_t n) -> Result<BigInt>` | The `n`-th Fibonacci number (`F_0 = 0`, `F_1 = 1`) via the **fast-doubling** identities `F(2k) = F(k)(2F(k+1) - F(k))` and `F(2k+1) = F(k+1)² + F(k)²` — `O(log n)` `BigInt` multiplies, not `O(n)` additions. `domain_error` for `n < 0`. `F(93)` and beyond (which overflow `int64`) are exact here. |
| `lucas` | `[[nodiscard]] auto lucas(std::int64_t n) -> Result<BigInt>` | The `n`-th Lucas number (`L_0 = 2`, `L_1 = 1`, `L_n = L_{n-1} + L_{n-2}`), iterative. `domain_error` for `n < 0`. Unbounded. |
| `bell` | `[[nodiscard]] auto bell(std::int64_t n) -> Result<BigInt>` | The `n`-th Bell number `B_n` (number of set partitions of `n` elements), computed via the Bell triangle. `domain_error` for `n < 0`. Exact and unbounded. |
| `partition_count` | `[[nodiscard]] auto partition_count(std::int64_t n) -> Result<BigInt>` | The number of **integer partitions** `p(n)` — ways to write `n` as an unordered sum of positive integers (`p(0) = 1`, `p(5) = 7`, `p(10) = 42`), via Euler's pentagonal-number recurrence `p(m) = Σ_{j≥1} (−1)^{j−1}[p(m − j(3j−1)/2) + p(m − j(3j+1)/2)]`. `domain_error` for `n < 0`. Exact, unbounded, super-polynomial growth. |
| `euler_number` | `[[nodiscard]] auto euler_number(std::int64_t n) -> Result<BigInt>` | The `n`-th **Euler number** `E_n` — the **secant** numbers `sec x = Σ E_{2n} x^{2n}/(2n)!`, a **signed** integer: `E_0 = 1`, `E_2 = −1`, `E_4 = 5`, `E_6 = −61`, …, with every odd-index `E_{2k+1} = 0`. Built from the boustrophedon (Seidel/Entringer) triangle of zigzag numbers `A_n`, then `E_n = (−1)^{n/2} A_n` for even `n`. `domain_error` for `n < 0`. |
| `subfactorial` | `[[nodiscard]] auto subfactorial(std::int64_t n) -> Result<BigInt>` | The **subfactorial** `!n` — the number of derangements of `n` elements (permutations with no fixed point) — via `!n = n·!(n−1) + (−1)^n` with `!0 = 1` (so `!1 = 0`, `!2 = 1`, `!3 = 2`, `!4 = 9`). `domain_error` for `n < 0`. Exact and unbounded (`!n ~ n!/e`). |

## Sequences of two indices

| Function | Signature | Meaning & domain |
| :--- | :--- | :--- |
| `binomial` | `[[nodiscard]] auto binomial(std::int64_t n, std::int64_t k) -> Result<BigInt>` | `C(n, k) = n!/(k!(n-k)!)`. Returns `0` when `k < 0` or `k > n` (the conventional empty count); `domain_error` for `n < 0`. Multiplicative formula `∏_{i=1..k}(n-k+i)/i` with an exact divide after each multiply. |
| `falling_factorial` | `[[nodiscard]] auto falling_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt>` | The falling factorial `n^(k) = n(n-1)…(n-k+1)`, a product of exactly `k` descending factors (empty product `1` when `k = 0`). Requires `k >= 0`, else `domain_error`; `n` is **unrestricted**, so a negative `n` yields a signed result. |
| `rising_factorial` | `[[nodiscard]] auto rising_factorial(std::int64_t n, std::int64_t k) -> Result<BigInt>` | The rising factorial `n^(k)- = n(n+1)…(n+k-1)`, a product of exactly `k` ascending factors (empty product `1` when `k = 0`). Requires `k >= 0`, else `domain_error`; `n` is unrestricted. |
| `stirling_second` | `[[nodiscard]] auto stirling_second(std::int64_t n, std::int64_t k) -> Result<BigInt>` | Stirling number of the second kind `S(n, k)` — the number of partitions of `n` labelled elements into `k` non-empty unlabelled subsets, via `S(n, k) = k·S(n-1, k) + S(n-1, k-1)`. Returns `0` when `k > n`; `domain_error` for `n < 0` or `k < 0`. Exact and unbounded. |
| `stirling_first_unsigned` | `[[nodiscard]] auto stirling_first_unsigned(std::int64_t n, std::int64_t k) -> Result<BigInt>` | Unsigned Stirling number of the first kind `c(n, k)` — the number of permutations of `n` elements with exactly `k` disjoint cycles, via `c(n, k) = (n-1)·c(n-1, k) + c(n-1, k-1)`. Returns `0` when `k > n`; `domain_error` for `n < 0` or `k < 0`. Exact and unbounded. |
| `eulerian_number` | `[[nodiscard]] auto eulerian_number(std::int64_t n, std::int64_t k) -> Result<BigInt>` | The **Eulerian number** `⟨n, k⟩` — the number of permutations of `{1..n}` with exactly `k` ascents, via `⟨n, k⟩ = (k+1)⟨n-1, k⟩ + (n-k)⟨n-1, k-1⟩` with `⟨0, 0⟩ = 1`. Returns `0` when `k >= n` (a permutation of `n >= 1` elements has at most `n-1` ascents); `domain_error` for `n < 0` or `k < 0`. Exact and unbounded — the row sum `Σ_k ⟨n, k⟩` is `n!`. |

## Multinomial coefficient

| Function | Signature | Meaning & domain |
| :--- | :--- | :--- |
| `multinomial` | `[[nodiscard]] auto multinomial(std::span<const std::int64_t> ks) -> Result<BigInt>` | The multinomial coefficient `(Σ k_i)! / ∏(k_i!)` — the number of distinct orderings of a multiset with the given multiplicities. All `k_i` must be `>= 0`, else `domain_error`. An **empty span yields `1`** (the empty product). Each factorial division is exact. |

`multinomial` sums the multiplicities into an `int64` accumulator: if that sum
`Σ k_i` would itself overflow `int64`, the call returns `domain_error` (the sum
must be representable to name `(Σ k_i)!`). The individual `BigInt` factorials it
then forms are, as everywhere in this module, unbounded.

## Error model

| Condition | Error |
| :--- | :--- |
| Negative index (`n < 0`) on `factorial`, `catalan`, `fibonacci`, `lucas`, `bell`, `binomial`, `stirling_second`, `stirling_first_unsigned`, `partition_count`, `euler_number`, `subfactorial`, `eulerian_number` | `MathError::domain_error` |
| `k < 0` on `falling_factorial` / `rising_factorial` | `MathError::domain_error` |
| `k < 0` (either index) on `stirling_second` / `stirling_first_unsigned` / `eulerian_number` | `MathError::domain_error` |
| `double_factorial` with `n < -1` | `MathError::domain_error` |
| `multinomial` with any `k_i < 0` | `MathError::domain_error` |
| `catalan` argument overflow (`2n`/`n+1` leaves `int64`) | `MathError::domain_error` |
| `multinomial` when `Σ k_i` overflows `int64` | `MathError::domain_error` |
| Exactness invariant violated (a would-be inexact interleaved division) | `MathError::undefined_value` — cannot arise in a correct build |

Note the differences from the `int64` module: `binomial` returns `0` (not an
error) for `k < 0` or `k > n`, exactly as its `int64` sibling does, and **there
is no `MathError::overflow` anywhere in this module** — the `BigInt` payload
cannot overflow. The only overflow-flavoured failures are the `int64`
*argument* guards on `catalan` and `multinomial`, and those surface as
`domain_error`.

## Worked examples

```cpp
import nimblecas.bigcombinatorics;
import nimblecas.bigint;
import nimblecas.core;
using namespace nimblecas;

// Factorials beyond the int64 ceiling stay exact (21! already overflows int64).
factorial(20).value().to_string();  // "2432902008176640000"  (largest int64 factorial)
factorial(25).value().to_string();  // "15511210043330985984000000"
factorial(-1).error();              // MathError::domain_error

// Binomial: multiplicative, divide-as-you-go; the conventional empty counts are 0.
binomial(52, 5).value().to_string();     // "2598960"
binomial(5, 7).value().to_string();      // "0"   (k > n)
binomial(5, -1).value().to_string();     // "0"   (k < 0)
binomial(100, 50).value().to_string();   // "100891344545564193334812497256"
// Symmetry and Pascal's rule hold on beyond-int64 arguments.
binomial(100, 30).value() == binomial(100, 70).value();  // true
binomial(200, 100).value() ==
    binomial(199, 99).value().add(binomial(199, 100).value());  // true
binomial(-1, 0).error();                 // MathError::domain_error

// Multinomial: (Σ k_i)! / ∏(k_i!); an empty multiset is the empty product 1.
const std::array<std::int64_t, 3> mix{2, 3, 4};
multinomial(mix).value().to_string();                     // "1260"  (= 9!/(2!3!4!))
multinomial(std::span<const std::int64_t>{}).value().to_string();  // "1"
const std::array<std::int64_t, 2> bad{-1, 2};
multinomial(bad).error();                                 // MathError::domain_error

// Catalan: exact where the int64 module overflows (from n = 36).
catalan(20).value().to_string();         // "6564120420"
catalan(-1).error();                     // MathError::domain_error

// Falling / rising factorial: k >= 0 required, n unrestricted (signed result allowed).
falling_factorial(5, 3).value().to_string();    // "60"   (5·4·3)
falling_factorial(-1, 3).value().to_string();   // "-6"   ((-1)(-2)(-3))
rising_factorial(2, 4).value().to_string();     // "120"  (2·3·4·5)
rising_factorial(7, 5).value() ==
    falling_factorial(11, 5).value();           // true   (rising(n,k) == falling(n+k-1,k))
falling_factorial(5, -1).error();               // MathError::domain_error

// Double factorial with the empty-product convention.
double_factorial(-1).value().to_string();       // "1"
double_factorial(9).value().to_string();        // "945"  (9·7·5·3·1)
double_factorial(-2).error();                   // MathError::domain_error

// Fibonacci by fast doubling; F(93) and beyond overflow int64 but are exact here.
fibonacci(100).value().to_string();  // "354224848179261915075"
fibonacci(200).value().to_string();  // "280571172992510140037611932413038677189525"
fibonacci(-1).error();               // MathError::domain_error

// Lucas numbers and the identity L(n) == 2·F(n+1) - F(n).
lucas(50).value().to_string();       // "28143753123"
lucas(40).value() ==
    fibonacci(41).value().add(fibonacci(41).value()).subtract(fibonacci(40).value());  // true

// Stirling numbers and Bell numbers, and the row-sum identities that tie them together.
stirling_second(5, 3).value().to_string();          // "25"
stirling_second(3, 5).value().to_string();          // "0"   (k > n)
stirling_first_unsigned(4, 2).value().to_string();  // "11"
bell(10).value().to_string();                        // "115975"
// sum_k S(10, k) == Bell(10)
BigInt row_sum{};
for (std::int64_t k = 0; k <= 10; ++k) {
    row_sum = row_sum.add(stirling_second(10, k).value());
}
row_sum == bell(10).value();                         // true

// Integer partitions p(n) via Euler's pentagonal recurrence.
partition_count(10).value().to_string();   // "42"
partition_count(100).value().to_string();  // "190569292"
partition_count(-1).error();               // MathError::domain_error

// Euler (secant) numbers: signed, odd-index entries vanish.
euler_number(0).value().to_string();       // "1"
euler_number(2).value().to_string();       // "-1"
euler_number(4).value().to_string();       // "5"
euler_number(6).value().to_string();       // "-61"
euler_number(3).value().to_string();       // "0"   (odd index)

// Subfactorial / derangements !n; the identity n! == sum_k C(n,k) !k holds beyond int64.
subfactorial(4).value().to_string();       // "9"
subfactorial(10).value().to_string();      // "1334961"

// Eulerian numbers ⟨n,k⟩ and the row-sum identity sum_k ⟨n,k⟩ == n!.
eulerian_number(3, 1).value().to_string();  // "4"    (⟨3,0..2⟩ = 1, 4, 1)
eulerian_number(4, 1).value().to_string();  // "11"   (⟨4,0..3⟩ = 1, 11, 11, 1)
eulerian_number(3, 5).value().to_string();  // "0"    (k >= n)
BigInt euler_row{};
for (std::int64_t k = 0; k <= 5; ++k) {
    euler_row = euler_row.add(eulerian_number(6, k).value());
}
euler_row == factorial(6).value();          // true   (== 720)
```

## See also

- [`nimblecas.combinatorics`](combinatorics.md) — the `int64`, overflow-checked
  sibling this module mirrors function-for-function (and the home of the
  `Rational`-valued `bernoulli()` that is deliberately absent here).
- `nimblecas.bigint` — the arbitrary-precision integer every count is carried
  in; its `divmod` backs the exact interleaved divisions.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway these free
  functions return on.
- [Documentation hub](../Index.md)
```
