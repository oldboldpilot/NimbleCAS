# `nimblecas.probmethod` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/probmethod/probmethod.cppm`

Exact, **sound** decision procedures for the **probabilistic method** (ROADMAP
§7.18, after Alon & Spencer's *The Probabilistic Method*): non-constructive
existence proofs that show a combinatorial object with a desired property exists
by proving a random construction has it with **positive probability**. Each entry
point takes the exact numeric ingredients of a classical argument — an
expectation, a list of bad-event probabilities, a variance, a dependency degree —
and returns a **verdict** together with the **exact rational (or BigInt) quantity**
the verdict rests on, so the caller can audit it independently.

This module is the *existence layer* built **on top of** the tail inequalities of
[`nimblecas.probdist`](probdist.md) (Markov / Chebyshev supply the analytic
justification); it does not re-derive them. Everything is exact over the rationals
`Q` wherever representable; the one transcendental constant that appears — Euler's
number `e`, in the Lovász Local Lemma — is handled by a **rigorous rational
enclosure**, never a fabricated decimal (see the honesty boundary below).

```cpp
import nimblecas.probmethod;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (`Rational`),
[`bigint`](bigint.md) (`BigInt`), and
[`bigcombinatorics`](bigcombinatorics.md) (BigInt binomials for the Ramsey bound).

## Verdicts

```cpp
enum class Existence : std::uint8_t { exists, not_certified };
enum class LovaszVerdict : std::uint8_t { exists, condition_fails, indeterminate };
```

- **`Existence::exists`** is a *sound* non-constructive proof that a configuration
  with the desired property is present.
- **`Existence::not_certified`** means the sufficient hypothesis of *this* argument
  is not met. It is **never** a claim that no good configuration exists — only that
  this particular argument is silent.
- The Lovász Local Lemma needs a third state because its threshold involves `e`:
  **`condition_fails`** (the LLL hypothesis provably does *not* hold) is
  distinguished from **`indeterminate`** (the exact constraint lands in the thin
  rational sliver around `1/e` that the enclosure cannot resolve).

`to_string_view(Existence)` and `to_string_view(LovaszVerdict)` render the verdicts.

## The existence procedures

### First-moment (expectation) argument

```cpp
[[nodiscard]] auto first_moment_exists(const Rational& expected_value) -> Result<FirstMoment>;

struct FirstMoment {
    Existence verdict;             // exists  iff  expected_value < 1
    Rational expected_value;       // E[X], echoed back
    bool attains_at_least_mean;    // ∃ω: X(ω) ≥ E[X]  (averaging; always true)
    bool attains_at_most_mean;     // ∃ω: X(ω) ≤ E[X]  (averaging; always true)
};
```

For a non-negative **integer** counting variable `X` (the number of "bad" events
that occur), if `E[X] < 1` then `P(X = 0) > 0`, so an outcome with `X = 0` — a
configuration avoiding **every** bad event — exists. The `attains_at_least_mean` /
`attains_at_most_mean` flags record the *averaging argument*: since the mean is a
weighted average of the attained values, some outcome is `≥ E[X]` and some is
`≤ E[X]`. Both hold unconditionally (independent of the `E[X] < 1` test).
`expected_value` must be `≥ 0` (`domain_error` otherwise).

### Union bound

```cpp
[[nodiscard]] auto union_bound_exists(std::span<const Rational> probs) -> Result<UnionBound>;

struct UnionBound {
    Existence verdict;             // exists  iff  total_probability < 1
    Rational total_probability;    // Σ_i P(bad_i), exact
};
```

If the exact summed probability of the bad events is `< 1`, then with positive
probability none occurs, so a good configuration exists. Each entry must be a
genuine probability in `[0, 1]` (`domain_error` otherwise); an empty span sums to
`0` and is vacuously `exists`. Propagates `MathError::overflow` if the exact
rational sum leaves `int64`.

### Second-moment method

```cpp
[[nodiscard]] auto second_moment_positive(const Rational& mean, const Rational& variance)
    -> Result<SecondMoment>;

struct SecondMoment {
    Existence verdict;             // exists  iff  prob_zero_bound < 1
    Rational prob_zero_bound;      // the exact bound  P(X = 0) ≤ Var/E[X]^2
};
```

By Chebyshev (via [`probdist`](probdist.md)), `P(X = 0) ≤ Var(X)/E[X]^2`. If that
ratio is `< 1` — equivalently `Var(X) < E[X]^2` — then `P(X > 0) > 0`, so an
outcome with `X > 0` (a configuration realising at least one desired object)
exists. `mean` must be `> 0` (`domain_error` for `mean ≤ 0`, which also guards the
`E[X] = 0` division) and `variance` must be `≥ 0`. The exact `Var / E[X]^2`
arithmetic propagates `overflow`.

### Lovász Local Lemma — symmetric

```cpp
[[nodiscard]] auto lovasz_symmetric(const Rational& p, std::int64_t d)
    -> Result<LovaszSymmetric>;

struct LovaszSymmetric {
    LovaszVerdict verdict;
    Rational constraint;           // p·(d+1); the LLL asks whether e·constraint ≤ 1
};

[[nodiscard]] auto e_lower_bound() -> Rational;   // 2718281/1000000  <  e
[[nodiscard]] auto e_upper_bound() -> Rational;   // 2718282/1000000  >  e
```

The symmetric LLL: if each bad event has probability `≤ p`, depends on at most `d`
others, and `e·p·(d+1) ≤ 1`, then **all** bad events are simultaneously avoidable,
so a good configuration exists. `p` must lie in `[0, 1]` and `d` must be `≥ 0`
(`domain_error` otherwise); `overflow` if the exact `p·(d+1)` leaves `int64`. This
is **the honesty crux** of the module — see below.

### Lovász Local Lemma — asymmetric (general)

```cpp
[[nodiscard]] auto lovasz_asymmetric(std::span<const Rational> probs,
                                     const std::vector<std::vector<int>>& dependency,
                                     std::span<const Rational> x_weights)
    -> Result<LovaszAsymmetric>;

struct LovaszAsymmetric {
    Existence verdict;             // exists  iff  every per-event inequality holds
    std::vector<Rational> slack;   // per-event  x_i·Π(1−x_j) − P(A_i)
    std::size_t first_violation;   // first failing index, or probs.size() if none
};
```

The general LLL with per-event weights `x_i ∈ (0, 1)`: the weights certify
simultaneous avoidability **iff**, for every event `i`,

```
P(A_i) ≤ x_i · Π_{j ~ i} (1 − x_j).
```

This is a pure **exact-rational** inequality check — no `e` appears. `probs[i] =
P(A_i) ∈ [0, 1]`; `dependency[i]` lists the neighbours `j ~ i` (indices in
`[0, n)`, `j ≠ i`) that `A_i` depends on; `x_weights[i] = x_i` must lie **strictly**
inside `(0, 1)`. `slack[i]` is the exact margin `x_i·Π(1−x_j) − P(A_i)` (`≥ 0`
exactly where event `i`'s inequality holds), and `first_violation` is the index of
the first event that fails, or `probs.size()` when all pass. `domain_error` on a
size mismatch, an out-of-range neighbour, a probability outside `[0, 1]`, or a
weight outside `(0, 1)`; `overflow` from the exact product/difference arithmetic.

### Ramsey lower bounds (first moment)

```cpp
[[nodiscard]] auto ramsey_certifies(std::int64_t n, std::int64_t k) -> Result<bool>;
[[nodiscard]] auto ramsey_lower_bound(std::int64_t k) -> Result<RamseyLowerBound>;

struct RamseyLowerBound {
    std::int64_t k;
    std::int64_t largest_n;        // R(k,k) > largest_n
    BigInt threshold;              // 2^{C(k,2) − 1}
};
```

A uniformly random 2-colouring of the edges of `K_n` has expected number of
monochromatic `K_k` equal to `C(n,k) · 2^{1 − C(k,2)}`. When that is `< 1` —
equivalently the exact BigInt comparison `C(n,k) < 2^{C(k,2) − 1}` — some colouring
has none, so `R(k,k) > n`. `ramsey_certifies(n, k)` returns whether this holds for a
given `(n, k)`; `ramsey_lower_bound(k)` returns the **largest** such `n`. Both use
exact BigInt binomials (from [`bigcombinatorics`](bigcombinatorics.md)) and the
exact BigInt `2^{C(k,2)−1}`, so they never overflow. Both require `k ≥ 2`
(`domain_error` otherwise), and `ramsey_certifies` requires `n ≥ 0`. The
`ramsey_lower_bound` search cost grows with `k` (the certified `n` grows roughly
exponentially in `k`).

Worked values (hand-verified): for `k = 3`, threshold `2^{C(3,2)-1} = 2^2 = 4`, and
since `C(3,3) = 1 < 4` but `C(4,3) = 4` is not, `largest_n = 3` (so `R(3,3) > 3`).
For `k = 4`, threshold `2^{C(4,2)-1} = 2^5 = 32`, and since `C(6,4) = 15 < 32` but
`C(7,4) = 35` is not, `largest_n = 6` (so `R(4,4) > 6`).

## Honesty boundary — the rigorous rational enclosure of `e`

The symmetric LLL certifies avoidability when `e · p · (d+1) ≤ 1`, where **`e` is
transcendental and therefore not representable in `Q`**. Comparing against a
truncated decimal "`e`" would silently make the verdict unsound — exactly the
plausible-but-wrong outcome Rule 32 forbids for an existence *certificate*.

Instead the module carries a **rigorous rational enclosure** `e_lo < e < e_hi`,
with

```
e_lo = 2718281/1000000 = 2.718281 < e = 2.718281828459045... < 2.718282 = 2718282/1000000 = e_hi,
```

both bounds proven from the decimal expansion of `e` (itself a rigorous truncation
of the convergent series `e = Σ 1/n!`). The verdict on the exact rational
constraint `c = p·(d+1)` is decided three ways:

| Condition (exact rational) | Implies | Verdict |
| :--- | :--- | :--- |
| `c · e_hi ≤ 1` | `e·c < e_hi·c ≤ 1`, so `e·p·(d+1) < 1` **genuinely** | `exists` (sound) |
| `c · e_lo > 1` | `e·c > e_lo·c > 1`, so the LLL hypothesis provably **fails** | `condition_fails` |
| otherwise | `1/e_hi < c ≤ 1/e_lo`: the enclosure cannot resolve `e·c` vs `1` | `indeterminate` |

Because `1/e` is irrational, **no rational `c` can hit the boundary exactly**, so
the `indeterminate` band is only as wide as `(e_hi − e_lo)` (here `10⁻⁶`); tighten
the enclosure to shrink it. Consequently **a positive certificate (`exists`) is
always sound** — a certified case genuinely satisfies `e·p·(d+1) < 1`. The two
bounds are exposed as `e_lower_bound()` / `e_upper_bound()` so callers (and tests)
can reconstruct and audit the soundness witness `constraint · e_upper_bound() ≤ 1`.

All rational comparisons throughout the module are exact: canonical `Rational`s
(positive denominators) are compared by `__int128` cross-multiplication, so no
`int64` product is ever formed and truncated. Where an exact quantity would leave
`int64`, the overflow-checked `Rational` arithmetic surfaces `MathError::overflow`
and this module propagates it rather than wrapping.

## API summary

| Name | Signature | Returns |
| :--- | :--- | :--- |
| `first_moment_exists` | `auto first_moment_exists(const Rational& expected_value) -> Result<FirstMoment>` | Expectation argument: `exists` iff `E[X] < 1`. |
| `union_bound_exists` | `auto union_bound_exists(std::span<const Rational> probs) -> Result<UnionBound>` | Union bound: `exists` iff `Σ P(bad_i) < 1`. |
| `second_moment_positive` | `auto second_moment_positive(const Rational& mean, const Rational& variance) -> Result<SecondMoment>` | Chebyshev bound `P(X=0) ≤ Var/E²`; `exists` iff `< 1`. |
| `lovasz_symmetric` | `auto lovasz_symmetric(const Rational& p, std::int64_t d) -> Result<LovaszSymmetric>` | Symmetric LLL via the rational `e`-enclosure. |
| `lovasz_asymmetric` | `auto lovasz_asymmetric(std::span<const Rational> probs, const std::vector<std::vector<int>>& dependency, std::span<const Rational> x_weights) -> Result<LovaszAsymmetric>` | General LLL: per-event inequality check + slack. |
| `ramsey_certifies` | `auto ramsey_certifies(std::int64_t n, std::int64_t k) -> Result<bool>` | Whether `C(n,k) < 2^{C(k,2)−1}`. |
| `ramsey_lower_bound` | `auto ramsey_lower_bound(std::int64_t k) -> Result<RamseyLowerBound>` | Largest `n` with `R(k,k) > n`. |
| `e_lower_bound` | `auto e_lower_bound() -> Rational` | `2718281/1000000 < e` (rigorous). |
| `e_upper_bound` | `auto e_upper_bound() -> Rational` | `2718282/1000000 > e` (rigorous). |
| `to_string_view` | `auto to_string_view(Existence) -> std::string_view`; `auto to_string_view(LovaszVerdict) -> std::string_view` | Verdict names. |

## Error model

| Condition | Result |
| :--- | :--- |
| `first_moment_exists`, `expected_value < 0` | `MathError::domain_error`. |
| `union_bound_exists`, any probability `∉ [0, 1]` | `MathError::domain_error`. |
| `union_bound_exists`, exact sum overflows `int64` | `MathError::overflow`. |
| `second_moment_positive`, `mean ≤ 0` or `variance < 0` | `MathError::domain_error`. |
| `second_moment_positive`, exact `Var/E²` overflows `int64` | `MathError::overflow`. |
| `lovasz_symmetric`, `p ∉ [0, 1]` or `d < 0` | `MathError::domain_error`. |
| `lovasz_symmetric`, `d = INT64_MAX` or `p·(d+1)` overflows `int64` | `MathError::overflow`. |
| `lovasz_asymmetric`, size mismatch / neighbour out of range / `P ∉ [0,1]` / `x ∉ (0,1)` | `MathError::domain_error`. |
| `lovasz_asymmetric`, exact product/difference overflows `int64` | `MathError::overflow`. |
| `ramsey_certifies` / `ramsey_lower_bound`, `k < 2` (or `n < 0`) | `MathError::domain_error`. |
| Any success | the corresponding record with an exact quantity; verdicts are never plausible-but-wrong. |

## Example

```cpp
import nimblecas.probmethod;
import nimblecas.ratpoly;
using namespace nimblecas;

auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// First moment: E[X] = 1/2 < 1  =>  a bad-event-free configuration exists.
auto fm = first_moment_exists(rat(1, 2)).value();
bool ok = fm.verdict == Existence::exists;              // true

// Union bound over three bad events summing to 3/4 < 1.
std::vector<Rational> probs{rat(1, 4), rat(1, 4), rat(1, 4)};
auto ub = union_bound_exists(probs).value();
auto total = ub.total_probability;                      // 3/4, and verdict == exists

// Second moment: mean 10, variance 5  =>  P(X=0) <= 1/20 < 1  =>  X > 0 exists.
auto sm = second_moment_positive(Rational::from_int(10), Rational::from_int(5)).value();
auto bound = sm.prob_zero_bound;                        // 1/20, verdict == exists

// Symmetric LLL: p = 1/100, d = 3  =>  c = 1/25, and c * e_upper <= 1 soundly.
auto ll = lovasz_symmetric(rat(1, 100), 3).value();
bool avoidable = ll.verdict == LovaszVerdict::exists;   // true (sound certificate)
auto witness = ll.constraint.multiply(e_upper_bound()).value();  // 2718282/25000000 ≤ 1

// Asymmetric LLL: two mutually dependent events, x_i = 1/4, P(A_i) = 1/8 ≤ 3/16.
std::vector<std::vector<int>> dep{{1}, {0}};
std::vector<Rational> x{rat(1, 4), rat(1, 4)};
std::vector<Rational> pa{rat(1, 8), rat(1, 8)};
auto al = lovasz_asymmetric(pa, dep, x).value();
bool certified = al.verdict == Existence::exists;       // true; al.slack == {1/16, 1/16}

// Ramsey: the classic first-moment lower bounds.
auto r3 = ramsey_lower_bound(3).value();                // largest_n = 3  =>  R(3,3) > 3
auto r4 = ramsey_lower_bound(4).value();                // largest_n = 6  =>  R(4,4) > 6
```

## See also

- [`nimblecas.probdist`](probdist.md) — the tail inequalities (Markov, Chebyshev,
  Chernoff, Hoeffding, …) this existence layer builds on.
- [`nimblecas.bigcombinatorics`](bigcombinatorics.md) — the exact BigInt binomials
  behind the Ramsey bound.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` used throughout.
- [Documentation hub](../Index.md)
```
