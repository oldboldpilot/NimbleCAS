# `nimblecas.pfd` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pfd/pfd.cppm`

Square-free **partial-fraction decomposition** of a rational function over the
rationals, `Q(x)`. Given `A(x)/B(x)` over `Q`, `partial_fractions` produces the
exact decomposition

```
A(x)/B(x) = P(x) + sum_i sum_{j=1}^{e_i}  C_{i,j}(x) / b_i(x)^j ,
            deg C_{i,j} < deg b_i,
```

where `B = prod_i b_i^{e_i}` is the square-free factorization of the
(monic-normalised) denominator, the `b_i` are monic, square-free and pairwise
coprime, and `P` is the polynomial part (zero iff `A/B` is already proper). It is
the rational-arithmetic capstone of ROADMAP §7.17, built entirely on the exact
`Q[x]` substrate of [`ratpoly`](ratpoly.md).

```cpp
import nimblecas.pfd;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## Square-free, not irreducible

This is the single most important thing to understand about the output. The
decomposition is **square-free, not irreducible** (ROADMAP §7.17): each base
`b_i` is square-free but may still be **composite**, and the pipeline never
factors a square-free polynomial into its irreducible pieces (that would need a
full factorization algorithm). Concretely, factors are separated **only when they
have different multiplicities** in `B`.

- `1/(x^2 - x) = 1/(x(x - 1))` — both roots have multiplicity 1, so the
  square-free denominator is the **composite** `x^2 - x` itself. The two coprime
  linear factors are **not** split: the result is the single term
  `1/(x^2 - x)`, left whole.
- `1/(x^2(x - 1))` — here `x` has multiplicity 2 and `x - 1` has multiplicity 1.
  The differing multiplicities let the square-free stage separate them, giving
  `-1/x - 1/x^2 + 1/(x - 1)`.
- `(3x + 1)/(x^3 + x) = (3x + 1)/(x(x^2 + 1))` — the denominator is already
  square-free (multiplicity 1 throughout), so `x` and `x^2 + 1` stay merged: the
  result is the single term `(3x + 1)/(x^3 + x)`.

Splitting `x^2 - x` into `1/(x-1) - 1/x`, or `x^2 + 1` into complex-linear
factors, is the job of an irreducible factorizer, not this module.

## The overflow contract

Following the rest of the engine, every stage is **exact** and
**overflow-checked** (Rule 32): all arithmetic runs through the underlying
`Q[x]` operations, so an `int64` numerator or denominator that would overflow
surfaces as `MathError::overflow` rather than silently wrapping. A zero
denominator is `MathError::division_by_zero`. Because `Q` is a field, every
polynomial division taken along the way is exact and the failure modes are
exactly those two — the same contract as [`ratpoly`](ratpoly.md).

## The three-stage pipeline

`partial_fractions` first normalises the denominator to monic (folding its
leading constant into the numerator, `A/B == (A/lc)/(B/lc)`) and peels off the
polynomial part by Euclidean division: `Am = P*Bm + R` with `deg R < deg Bm`.
The proper remainder `R/Bm` is then decomposed in three exact stages.

### 1. Yun square-free factorization — `square_free_factorization`

Yun's algorithm factors the monic denominator `Bm = prod_i b_i^{e_i}` over `Q`
using only gcds with the derivative and exact divisions. It emits the distinct
square-free bases with their multiplicities, in **ascending multiplicity order**.
For `x^2(x - 1) = x^3 - x^2` it yields `(x - 1, 1)` then `(x, 2)`; for `(x-1)^3`
it yields the single pair `(x - 1, 3)`.

### 2. Bezout distinct-factor split

The pairwise-coprime prime powers `p_i = b_i^{e_i}` are separated by the extended
Euclidean algorithm. For each factor the split solves the Bezout identity
`s*p + t*q = 1` for the coprime pair `(p, q)` where `q = Bm/p` (accumulated as a
suffix product of the remaining factors), then reduces to proper numerators via

```
R/(p*q) = (R*t)/p + (R*s)/q,   N_i = (R*t) mod p,   R' = (R*s) mod q.
```

This yields `R/Bm = sum_i N_i / p_i` with `deg N_i < deg p_i` and each `N_i`
sitting over a single prime power `b_i^{e_i}`. Non-coprime inputs would fail
`MathError::domain_error`, but by construction the square-free bases are coprime.

### 3. Base-`b` power expansion

Each `N_i / b_i^{e_i}` is spread across the ascending powers `b_i^1 .. b_i^{e_i}`
by a base-`b_i` expansion: repeatedly divide by `b_i` to read off the "digits"
`D_t` (low order first), where `C_{e-t} = D_t`. Because `deg N_i < e_i * deg b_i`
there are exactly `e_i` digits (the final quotient is zero), and each remainder
`C_{i,j}` has `deg < deg b_i`. Numerators that vanish are omitted from the output.

## Public API

```cpp
[[nodiscard]] auto square_free_factorization(const RationalPoly& f)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>>;

[[nodiscard]] auto partial_fractions(const RationalPoly& numerator,
                                     const RationalPoly& denominator)
    -> Result<PartialFraction>;

[[nodiscard]] auto to_string(const PartialFraction& pf, std::string_view var = "x")
    -> std::string;
```

### Result types

```cpp
// One summand C(x) / factor(x)^power of the proper part, with `factor` monic
// and square-free and deg C < deg factor.
struct PartialFractionTerm {
    RationalPoly factor;     // b_i — a monic, square-free base
    std::int64_t power;      // j >= 1 — the power of b_i in this summand
    RationalPoly numerator;  // C_{i,j} — deg < deg factor
};

// A(x)/B(x) = polynomial_part + sum over terms.
struct PartialFraction {
    RationalPoly polynomial_part;            // P(x); zero when A/B is proper
    std::vector<PartialFractionTerm> terms;  // the proper part, per prime power
};
```

### `square_free_factorization(f)`

Square-free factorization of `f` over `Q` via Yun's algorithm. Returns pairs
`(a_i, i)` with each `a_i` **monic**, square-free, of degree `>= 1`, the `a_i`
pairwise coprime, and `leading_coefficient(f) * prod_i a_i^i == f`. A zero or
constant `f` has no positive-degree factors and yields an **empty** list. Fails
`MathError::overflow` on an `int64` coefficient limit.

### `partial_fractions(numerator, denominator)`

Runs the full three-stage pipeline and returns a `PartialFraction`. Error model:

| Condition | Error |
| :--- | :--- |
| `denominator` is the zero polynomial | `MathError::division_by_zero` |
| an `int64` coefficient computation wraps | `MathError::overflow` |

The input need not be reduced: a common factor between numerator and denominator
decomposes correctly (e.g. `x/x^2` yields `1/x`). A constant denominator gives a
purely polynomial result (`terms` empty). The `polynomial_part` is zero exactly
when `A/B` is proper.

### `to_string(pf, var = "x")`

Human-readable rendering for diagnostics, e.g. `x + (1/2)/(x - 1) + (-1)/(x)^2`;
the zero decomposition renders as `"0"`. It is a debugging aid, not a canonical
or round-trippable form.

## Examples

```cpp
import nimblecas.pfd;
import nimblecas.ratpoly;
using namespace nimblecas;

// Build x^k-style inputs from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// Improper: x^3 / (x^2 - 1) = x + x/(x^2 - 1).
// Long division peels off P = x; the square-free remainder stays over x^2 - 1.
auto pf1 = partial_fractions(ipoly({0, 0, 0, 1}), ipoly({-1, 0, 1})).value();
// pf1.polynomial_part == x;  one term: numerator x over (x^2 - 1)^1.

// Repeated composite factor with linear numerators:
// x^3 / (x^2 + 1)^2 = x/(x^2 + 1) - x/(x^2 + 1)^2.
auto pf2 = partial_fractions(ipoly({0, 0, 0, 1}),
                             ipoly({1, 0, 2, 0, 1})).value();
// terms: (x)/(x^2 + 1)^1  and  (-x)/(x^2 + 1)^2.

// High multiplicity, base-b expansion past the second digit:
// x^2 / (x - 1)^3 = 1/(x - 1) + 2/(x - 1)^2 + 1/(x - 1)^3.
auto pf3 = partial_fractions(ipoly({0, 0, 1}), ipoly({-1, 3, -3, 1})).value();
// terms: 1/(x-1)^1, 2/(x-1)^2, 1/(x-1)^3.

// Square-free, not irreducible: 1/(x^2 - x) stays whole (both roots mult. 1).
auto pf4 = partial_fractions(ipoly({1}), ipoly({0, -1, 1})).value();
// single term: 1/(x^2 - x)^1 — x and (x - 1) are NOT separated.

// Differing multiplicities DO separate: 1/(x^2 (x - 1)).
auto pf5 = partial_fractions(ipoly({1}), ipoly({0, 0, -1, 1})).value();
// terms: (-1)/x^1, (-1)/x^2, (1)/(x - 1)^1.

// Zero denominator fails.
auto bad = partial_fractions(ipoly({1}), RationalPoly{});  // division_by_zero
```

## Relationship to integration

The square-free denominator factorization plus the Bezout identity `s*b_i + t*b_i' = 1`
is exactly the machinery **Hermite reduction** needs to compute the rational part
of `int A(x)/B(x) dx` without fully factoring `B` (ROADMAP §7.19). This module is
therefore the substrate for rational-function integration: Hermite reduction
reuses the square-free factorization and Bezout split here, and Rothstein–Trager
then handles the remaining square-free logarithmic part.

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder, monic Euclidean gcd, derivative) this
  module is built on.
- [`nimblecas.polynomial`](polynomial.md) — the integer ring `Z[x]` and its own
  Yun square-free factorization over `Z`.
- [Documentation hub](../Index.md)
