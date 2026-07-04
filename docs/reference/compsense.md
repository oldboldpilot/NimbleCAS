# `nimblecas.compsense` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/compsense/compsense.cppm`

Compressed sensing / sparse recovery (ROADMAP §7 algorithmics layer), split
along an **honest exact-vs-numerical boundary**. Two of the four recovery tools
are **exact over `Q`**: [`basis_pursuit`](#basis_pursuit--exact-over-q) solves
`min ||x||_1  s.t.  A x = b` as a linear program and returns the exact rational
minimiser — an answer reported as `p/q` is exactly `p/q`, nothing rounds — and
[`mutual_coherence_squared`](#mutual_coherence_squared) is an exact rational
diagnostic in its squared form. The other three —
[`orthogonal_matching_pursuit`](#orthogonal_matching_pursuit),
[`cosamp`](#cosamp), and
[`iterative_hard_thresholding`](#iterative_hard_thresholding) — are **numerical**
(`double`) greedy / thresholding heuristics that carry **no** global-optimality
guarantee: they recover a sparse signal only under sparsity / RIP / incoherence
conditions and can fail silently outside them. Recovery — exact or numerical — is
**never universal**; it holds only under sparsity / coherence conditions.

```cpp
import nimblecas.compsense;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`lp`](lp.md).

## How `basis_pursuit` stays exact

`basis_pursuit` solves `min ||x||_1  s.t.  A x = b` as a linear program and
returns the exact rational minimiser. Splitting `x = x⁺ − x⁻` (`x⁺, x⁻ ≥ 0`)
gives a standard-form **equality** LP with an RHS-signed `b`, which
[`nimblecas.lp`](lp.md)'s single-phase, origin-feasible Simplex (inequality
programs `A z <= b` with `b >= 0` only) cannot take directly. So this layer feeds
`lp` the exact **dual**,

```
maximize  b · y    subject to   |a_j · y| <= 1  for every atom a_j,   y free,
```

an inequality LP with an all-ones (hence `>= 0`) right-hand side — a perfect fit
for `maximize`. Strong duality gives the optimal L1 value, and the optimal `y*`
pins the support via **complementary slackness**: `x_j` can be nonzero only where
`|a_j · y*| = 1`, an **exact** rational equality test. The values on that support
are then recovered exactly by an exact-rational normal-equation solve
(`(A_S^T A_S) x_S = A_S^T b`) through [`nimblecas.matrix`](matrix.md). Everything
is over `Q`; nothing rounds.

**Honest caveat.** The support read-off is exact under the standard
basis-pursuit non-degeneracy / uniqueness regime (the regime L1 recovery is meant
for). Under LP degeneracy the tight set can be a **superset** of the true
support; the normal-equation solve still returns the exact minimiser as long as
the tight columns are linearly independent, and otherwise reports
`domain_error` rather than guessing.

## The overflow contract

Following the rest of the engine, the exact tools are **overflow-checked**
(Rule 32): every exact inner product, tightness test, and recovery solve flows
through `Rational`'s checked arithmetic (and the exact three-way compare
cross-multiplies under a guard), so an `int64` numerator or denominator that
would overflow surfaces as `MathError::overflow` rather than silently wrapping.
The numerical tools work in `double` and do not overflow-check; their only
reported failure is a shape mismatch (`domain_error`).

## Exact-rational API

All symbols live directly in namespace `nimblecas`.

### `basis_pursuit`

```cpp
[[nodiscard]] auto basis_pursuit(const std::vector<std::vector<Rational>>& A,
                                 const std::vector<Rational>& b)
    -> Result<std::vector<Rational>>;
```

The **exact** minimiser of `||x||_1` subject to `A x = b`, over `Q`. `A` is an
`m × n` rational matrix given as `m` rows of width `n` (the `n` atoms are its
**columns**); `b` has length `m`. Returns the length-`n` exact rational minimiser
`x`; off-support atoms stay exactly zero. A `b = 0` yields the zero vector.

### `mutual_coherence_squared`

```cpp
[[nodiscard]] auto mutual_coherence_squared(const std::vector<std::vector<Rational>>& A)
    -> Result<Rational>;
```

The **exact squared** mutual coherence of the dictionary whose atoms are the
columns of `A` (`m` rows × `n` columns):

```
max_{i != j}  <a_i, a_j>^2 / (||a_i||^2 ||a_j||^2).
```

Squaring avoids the `sqrt`, so the result is an exact rational in `[0, 1]`.

### `coherence_guarantees_recovery`

```cpp
[[nodiscard]] auto coherence_guarantees_recovery(const Rational& mu_squared, std::size_t k)
    -> Result<bool>;
```

The **exact** sufficient recovery test `k < (1 + 1/mu)/2`, evaluated without a
`sqrt`. For `k >= 1` and `mu > 0` the bound is equivalent to
`mu^2 (2k − 1)^2 < 1`, an exact rational comparison taking the squared coherence
directly. Returns `true` when the **sufficient — not necessary** — guarantee
holds: satisfying it guarantees OMP / basis-pursuit recovery of every `k`-sparse
signal; **violating it does not imply failure**.

## Numerical (double) API

Every measurement matrix is passed as a row-major `std::span<const double>` of
`rows*cols` entries: entry `(i, j)` lives at `data[i * cols + j]`, and its `n`
atoms are the **columns**. These are greedy / thresholding heuristics with **no**
global-optimality guarantee — correct only under sparsity / RIP / incoherence
conditions. A shape mismatch surfaces as `MathError::domain_error`.

### `orthogonal_matching_pursuit`

```cpp
[[nodiscard]] auto orthogonal_matching_pursuit(std::span<const double> A, std::size_t rows,
                                               std::size_t cols, std::span<const double> b,
                                               std::size_t sparsity, double tol = 1e-9)
    -> Result<std::vector<double>>;
```

**OMP.** Greedily selects, at each step, the atom most correlated with the
current residual (largest `|<a_j, r>|`), re-solves the least squares over the
whole active set against the original `b`, and updates the residual. Stops at
`sparsity` atoms or when the residual norm falls to `<= tol`. Returns the
length-`cols` coefficient vector.

### `cosamp`

```cpp
[[nodiscard]] auto cosamp(std::span<const double> A, std::size_t rows, std::size_t cols,
                          std::span<const double> b, std::size_t sparsity, int max_iter = 50,
                          double tol = 1e-9) -> Result<std::vector<double>>;
```

**CoSaMP** (Compressive Sampling Matching Pursuit). Each iteration forms the
signal proxy `A^T r`, merges its `2k` largest-magnitude coordinates with the
current support, least-squares-fits on that enlarged set, then prunes back to the
`k` largest. Stops on a small residual (`<= tol`) or after `max_iter` iterations.

### `iterative_hard_thresholding`

```cpp
[[nodiscard]] auto iterative_hard_thresholding(std::span<const double> A, std::size_t rows,
                                               std::size_t cols, std::span<const double> b,
                                               std::size_t sparsity, double step = 1.0,
                                               int max_iter = 500, double tol = 1e-9)
    -> Result<std::vector<double>>;
```

**IHT.** Iterates `x <- H_k( x + step · A^T (b − A x) )`, where `H_k` keeps the
`k` largest-magnitude coordinates and zeroes the rest. Stops when the update is
smaller than `tol` or after `max_iter` iterations.

### `mutual_coherence`

```cpp
[[nodiscard]] auto mutual_coherence(std::span<const double> A, std::size_t rows,
                                    std::size_t cols) -> Result<double>;
```

The **numerical** mutual coherence of the column-atom dictionary `A` (`m × n`):
the plain `max_{i != j} |<a_i, a_j>| / (||a_i|| ||a_j||)` as a `double` — the
`sqrt` of what `mutual_coherence_squared` returns exactly.

## Error model

| Condition | Error |
| :--- | :--- |
| `basis_pursuit`: empty `A`, zero-width rows, a ragged `A`, or a `b` whose length disagrees with `A` | `MathError::domain_error` |
| `basis_pursuit`: an **infeasible** system (`b` not in `range(A)`, certified by an unbounded dual) | `MathError::domain_error` |
| `basis_pursuit`: a degenerate optimum whose tight columns are dependent, so the exact support cannot be uniquely recovered | `MathError::domain_error` |
| `basis_pursuit`: an `int64` overflow in the exact tableau or the recovery solve | `MathError::overflow` |
| `mutual_coherence_squared`: an empty or ragged `A`, or fewer than two atoms (`n < 2`) | `MathError::domain_error` |
| `mutual_coherence_squared`: a zero-norm atom (its coherence is undefined) | `MathError::domain_error` |
| `mutual_coherence_squared`: an `int64` overflow in the exact inner products | `MathError::overflow` |
| `coherence_guarantees_recovery`: `k == 0`, or a negative `mu_squared` | `MathError::domain_error` |
| `coherence_guarantees_recovery`: an `int64` overflow in `mu^2 (2k − 1)^2` | `MathError::overflow` |
| `mutual_coherence`: fewer than two atoms (`cols < 2`), or a zero-norm atom | `MathError::domain_error` |
| Any numerical tool: `A.size() != rows * cols` or `b.size() != rows` | `MathError::domain_error` |
| OMP / CoSaMP: a singular active-set Gram matrix in the least-squares step | `MathError::domain_error` |

## Worked examples

From `tests/compsense_tests.cpp`:

```cpp
import nimblecas.compsense;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v)               { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// --- EXACT basis pursuit over Q --------------------------------------------------
// Atoms are the columns of A: a0=(1,0), a1=(1,1), a2=(0,1).
const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                           {ri(0), ri(1), ri(1)}};

// b = (2, 3): the unique min-L1 solution is the 2-sparse x = (0, 2, 1), EXACTLY.
basis_pursuit(A, {ri(2), ri(3)}).value();      // [0, 2, 1]   (no rounding)

// A genuinely fractional exact minimiser: b = (1/2, 5/2) => x = (0, 1/2, 2).
basis_pursuit(A, {rat(1, 2), rat(5, 2)}).value();  // [0, 1/2, 2]

// b = 0 => the exact minimiser is the zero vector.
basis_pursuit(A, {ri(0), ri(0)}).value();      // [0, 0, 0]

// --- EXACT coherence diagnostics -------------------------------------------------
// Columns (1,0), (1,1), (0,1): pairwise squared coherence is exactly 1/2.
mutual_coherence_squared(A).value();           // 1/2   (exact rational)

// With mu^2 = 1/2 the sufficient bound k < (1 + 1/mu)/2 holds for k=1, not k=2
// (it is sufficient, NOT necessary — k=2 may still recover in practice).
coherence_guarantees_recovery(rat(1, 2), 1).value();   // true
coherence_guarantees_recovery(rat(1, 2), 2).value();   // false
coherence_guarantees_recovery(rat(1, 100), 3).value(); // true  (very incoherent)

// --- NUMERICAL recovery (double, heuristic) --------------------------------------
// 4x5 incoherent dictionary: e0..e3 plus a4 = (1/2,1/2,1/2,1/2). Signal 3 e1 - 2 e3.
const std::vector<double> Ad{1, 0, 0, 0, 0.5,
                             0, 1, 0, 0, 0.5,
                             0, 0, 1, 0, 0.5,
                             0, 0, 0, 1, 0.5};
const std::vector<double> b{0.0, 3.0, 0.0, -2.0};

orthogonal_matching_pursuit(Ad, 4, 5, b, 2, 1e-9).value();  // ~ [0, 3, 0, -2, 0]
cosamp(Ad, 4, 5, b, 2, 50, 1e-9).value();                   // ~ [., 3, ., -2, .]

// IHT on an orthonormal (identity) dictionary recovers the 2-sparse signal.
const std::vector<double> I{1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1};
iterative_hard_thresholding(I, 4, 4, b, 2, 1.0, 500, 1e-12).value();  // ~ [0, 3, 0, -2]

// The numerical coherence is the sqrt of the exact squared value.
const std::vector<double> Ac{1, 1, 0, 0, 1, 1};
mutual_coherence(Ac, 2, 3).value();            // ~ sqrt(1/2) ~ 0.70710678

// --- Error model -----------------------------------------------------------------
basis_pursuit({{ri(1), ri(1), ri(0)}, {ri(1)}}, {ri(1), ri(1)}).error();
//   MathError::domain_error   (ragged A)
basis_pursuit(A, {ri(1)}).error();
//   MathError::domain_error   (b length mismatch)
mutual_coherence_squared({{ri(1)}, {ri(0)}}).error();
//   MathError::domain_error   (single atom — coherence undefined)
orthogonal_matching_pursuit({1.0, 0.0, 0.0}, 4, 5, b, 2, 1e-9).error();
//   MathError::domain_error   (data span length != rows*cols)
```

## See also

- [`nimblecas.lp`](lp.md) — the exact `Rational` Simplex `basis_pursuit` calls
  through its dual for exact L1 minimisation.
- [`nimblecas.matrix`](matrix.md) — exact dense linear algebra over `Q`; the
  normal-equation `solve` behind exact support recovery.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every exact
  quantity here lives in.
- [Documentation hub](../Index.md)
