# `nimblecas.bigeigen` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigeigen/bigeigen.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.eigen`](eigen.md).
Where `eigen` analyses an `int64`-[`Rational`](ratpoly.md) [`Matrix`](matrix.md)
and surfaces `MathError::overflow` (Rule 32) once an intermediate product or
result exceeds the `~9.2e18` `int64` ceiling, `bigeigen` works over a
[`BigMatrix`](bigmatrix.md) whose entries are `BigRational`s (arbitrary-precision
`BigInt` fractions). That removes the ceiling entirely: the spectrum of a matrix
whose entries — or whose Faddeev–LeVerrier intermediates — overflow `int64` stays
**exactly** the element of `Q` it mathematically is, with no rounding and no
bound on magnitude. This is the **slow-but-exact tier** that big-backs `eigen`.

The module answers two exact linear-algebra questions about a square
`BigMatrix` over `Q`, with two cheap byproducts:

1. **`characteristic_polynomial(A)`** — the coefficients of
   `p(x) = det(xI − A)`, computed **exactly** by the **Faddeev–LeVerrier**
   algorithm. That algorithm is the right fit for the big tier because every step
   it needs is one `BigMatrix` / `BigRational` already supports exactly: a matrix
   multiply, a trace (a sum of diagonal entries), and a division by a small
   positive integer `k` in `{1, …, n}`. No eigen-solver, no floating point, no
   symbolic determinant expansion — just the field operations of `Q`, each of
   which `BigRational` performs without overflow. The returned polynomial is
   **monic** and **low-degree-first** (coefficient of `x^j` at index `j`), size
   `n + 1`, matching the coefficient convention used across the engine.
2. **`rational_eigenvalues(A)`** — the **rational** eigenvalues with
   multiplicity, i.e. the rational roots of that characteristic polynomial, found
   via the **rational-root theorem + deflation** reimplemented directly over the
   `BigInt` / `BigRational` tier so it never overflows. (The `int64`
   [`roots`](roots.md) solver that `eigen` reuses would overflow on the large
   coefficients a big characteristic polynomial can produce, which is why the
   root search is rebuilt here on `BigInt`.)

The two byproducts of the same Faddeev–LeVerrier sweep are **`determinant(A)`**
`= (−1)^n · p(0)` and **`inverse(A)`** `= −M_n / c_n` (the final Faddeev matrix
scaled by the reciprocal of the constant term), the latter failing with
`domain_error` when `A` is singular.

```cpp
import nimblecas.bigeigen;
```

Depends on [`core`](core.md), `bigint`, `bigrational`,
[`bigmatrix`](bigmatrix.md), [`matrix`](matrix.md), and the shared
[`ratpoly`](ratpoly.md) coefficient convention.

## The honesty boundary

This boundary is true **in behaviour**, not just in the prose:

- **The characteristic polynomial IS the full, exact answer.** Every eigenvalue —
  rational, irrational, or complex — is a root of `p(x)`, and all `n + 1`
  coefficients are exact elements of `Q`.
- **`rational_eigenvalues` returns ONLY the rational eigenvalues.** Irrational
  eigenvalues (surds, e.g. the `±√2` of `[[0,1],[2,0]]`) and complex eigenvalues
  (e.g. the `±i` of the rotation `[[0,1],[-1,0]]`) are **not** extracted — there
  are no radicals, no `RootOf`, and no complex field in this layer. Such an
  eigenvalue simply does not appear in the returned list. **This is not a full
  eigendecomposition.** It is an exact *partial* answer: what is returned is
  provably correct, and what is omitted is omitted, never rounded.
- **The caller can detect an incomplete spectrum.** Sum the returned
  multiplicities and compare against `n`: a total equal to `n` means the spectrum
  is entirely rational; a total **below** `n` means some eigenvalues are
  irrational or complex and were (honestly) omitted.
- **Divisor enumeration is trial division over `BigInt`.** The rational-root
  search factors the cleared constant and leading coefficients by trial division
  up to `√|coeff|`, so it is exact but `O(√|coeff|)` `BigInt` operations. It is
  fast when those coefficients have small prime factors (the typical CAS case); a
  characteristic polynomial whose constant or leading term is a large,
  hard-to-factor semiprime is a **slow-but-correct** path, not an incorrect one.

## Free functions

All four entry points are free functions in `namespace nimblecas`, each
`[[nodiscard]]` and returning a `Result<T>`. All require a **square**
`BigMatrix`; a non-square input is rejected with `MathError::domain_error`.

```cpp
[[nodiscard]] auto characteristic_polynomial(const BigMatrix& a)
    -> Result<std::vector<BigRational>>;

[[nodiscard]] auto determinant(const BigMatrix& a) -> Result<BigRational>;

[[nodiscard]] auto inverse(const BigMatrix& a) -> Result<BigMatrix>;

[[nodiscard]] auto rational_eigenvalues(const BigMatrix& a)
    -> Result<std::vector<std::pair<BigRational, std::int64_t>>>;
```

### `characteristic_polynomial`

The exact `p(x) = det(xI − A)` as its `BigRational` coefficients
**low-degree-first**: the returned vector `r` has size `n + 1`, with `r[j]` the
coefficient of `x^j`, and is always **monic** (`r[n] == 1`). Computed by the
Faddeev–LeVerrier recurrence (`M_1 = I`, `c_0 = 1`; `c_k = −trace(A·M_k) / k`;
`M_{k+1} = A·M_k + c_k·I`), which reuses the single product `A·M_k` for both the
trace and the next Faddeev matrix, so each iteration performs only one matrix
multiply. Every division is by an integer `k ∈ {1, …, n}` — never zero — and is
exact over `Q`. The **`0×0`** matrix yields the constant polynomial `{1}` (its
empty-product characteristic polynomial). A non-square matrix yields
`domain_error`.

### `determinant`

The exact determinant as the Faddeev–LeVerrier byproduct
`det(A) = (−1)^n · p(0)`, where `p(0)` is the constant term (low-degree-first
index `0`) of the characteristic polynomial. It agrees with
[`BigMatrix::determinant`](bigmatrix.md) (which uses fraction-free Bareiss); both
are exact. The **`0×0`** determinant is the empty product `1`. A non-square
matrix yields `domain_error`.

### `inverse`

The exact inverse as the Faddeev–LeVerrier byproduct `A^{-1} = −M_n / c_n`, where
`M_n` is the final Faddeev matrix and `c_n = p(0)` is the constant term. A
**singular** matrix (`c_n == 0`, equivalently `det(A) == 0`) has no inverse and
yields `MathError::domain_error`; a non-square matrix likewise yields
`domain_error`. The **`0×0`** matrix inverts to the `0×0` matrix.

### `rational_eigenvalues`

The **rational** eigenvalues of `A`, each paired with its multiplicity (`≥ 1`) as
a `std::pair<BigRational, std::int64_t>`, **sorted ascending by eigenvalue**
(`BigRational` is a total order). These are exactly the rational roots of the
characteristic polynomial, found by the rational-root theorem over
`BigInt` / `BigRational`: denominators are cleared to integer coefficients, the
`±u/v` candidates are enumerated from the divisors of the lowest nonzero
coefficient (`u`) and the leading coefficient (`v`) — plus an explicit `0`
candidate when the constant term vanishes — and each candidate is tested by
synthetic division against a shrinking work polynomial, deflating confirmed roots
to recover **algebraic** multiplicity. Deflating on the work polynomial makes
duplicate candidates (e.g. `2/1` and `4/2`) harmless.

Per the honesty boundary, irrational and complex eigenvalues are **not**
returned, so the sum of the reported multiplicities is `n` iff the spectrum is
entirely rational (and is `< n` otherwise). The **`0×0`** matrix yields an empty
list; a non-square matrix yields `domain_error`. A matrix with no rational
eigenvalues at all is **not** an error — it returns an empty vector.

## Error model

| Condition | Error |
| :--- | :--- |
| `characteristic_polynomial` / `determinant` / `inverse` / `rational_eigenvalues` on a non-square matrix | `MathError::domain_error` |
| `inverse` of a singular matrix (`det(A) == 0`) | `MathError::domain_error` |

There is **no `overflow` row**: `BigRational` arithmetic cannot wrap, so the
`overflow` that the `int64` [`eigen`](eigen.md) can raise is unreachable here.
Every internal division (the Faddeev step `/ k`, the denominator-clearing LCM
divides, the divisor `divmod`) has a provably nonzero divisor, so a
divide-by-zero cannot fire; `Result` is threaded only for a uniform, defensive
surface. A matrix with no rational eigenvalues, and the `0×0` matrix, are valid
inputs, not errors.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigmatrix;
import nimblecas.bigeigen;
using namespace nimblecas;

// Exact BigRational helpers: the integer v as v/1, the fraction n/d, and a
// decimal string for magnitudes beyond int64.
auto bi   = [](std::int64_t v) { return BigRational::from_int(v); };
auto brat = [](std::int64_t n, std::int64_t d) {
    return BigRational::make(BigInt::from_i64(n), BigInt::from_i64(d)).value();
};
auto brs  = [](std::string_view s) { return BigRational::from_string(s).value(); };

// Build a BigMatrix from integer rows.
auto bmat = [](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<BigRational>> r;
    for (const auto& row : rows) {
        std::vector<BigRational> rr;
        for (std::int64_t v : row) rr.push_back(BigRational::from_int(v));
        r.push_back(std::move(rr));
    }
    return BigMatrix::from_rows(std::move(r)).value();
};

// Sum of the returned multiplicities: == n iff the spectrum is fully rational.
auto total_mult = [](const std::vector<std::pair<BigRational, std::int64_t>>& e) {
    std::int64_t s = 0; for (const auto& [v, m] : e) s += m; return s;
};

// --- Characteristic polynomial + a fully rational spectrum ------------------
// det(xI − diag(2,3,5)) = (x−2)(x−3)(x−5) = x^3 − 10x^2 + 31x − 30.
// Low-degree-first, monic, size n+1 = 4: [−30, 31, −10, 1].
auto d = bmat({{2, 0, 0}, {0, 3, 0}, {0, 0, 5}});
characteristic_polynomial(d).value();   // {−30, 31, −10, 1}  (r.back() == 1)

// Its rational eigenvalues are {2, 3, 5}, each simple, sorted ascending.
auto eig = rational_eigenvalues(d).value();
eig.size();            // 3
eig.front().first == bi(2) && eig.back().first == bi(5);  // ascending
total_mult(eig) == 3;  // fully rational (== n)

// Edge conventions: 1×1 → x − 7, and the 0×0 char-poly is the constant 1.
characteristic_polynomial(bmat({{7}})).value();          // {−7, 1}
characteristic_polynomial(BigMatrix::identity(0)).value();  // {1}
rational_eigenvalues(BigMatrix::identity(0)).value().empty();  // true

// --- A repeated root: algebraic multiplicity 2 -----------------------------
// Jordan block [[2,1],[0,2]] → (x−2)^2: one eigenvalue 2 of multiplicity 2.
auto j = rational_eigenvalues(bmat({{2, 1}, {0, 2}})).value();
j.size();              // 1  (one distinct eigenvalue)
j[0].first == bi(2) && j[0].second == 2;  // (2, 2)
total_mult(j) == 2;    // multiplicities sum to n = 2

// --- Honesty case 1: a purely complex spectrum -----------------------------
// Rotation [[0,1],[-1,0]] has eigenvalues ±i: charpoly x^2 + 1, NO rational
// roots, so rational_eigenvalues honestly returns an empty list.
auto rot = bmat({{0, 1}, {-1, 0}});
characteristic_polynomial(rot).value();      // {1, 0, 1}  (x^2 + 1)
auto rot_eig = rational_eigenvalues(rot).value();
rot_eig.empty();               // true
total_mult(rot_eig) == 0;      // 0 < n = 2: spectrum NOT fully rational

// --- Honesty case 2: a MIXED spectrum, 0 < total_mult < n ------------------
// block-diag([1], [[0,1],[2,0]]) has spectrum {1, +√2, −√2}: (x−1)(x^2−2).
// The one rational eigenvalue is found; the two surds are omitted, and the
// caller can tell the spectrum is only PARTLY rational.
auto mix = bmat({{1, 0, 0}, {0, 0, 1}, {0, 2, 0}});
characteristic_polynomial(mix).value();      // {2, −2, −1, 1}  (x^3 − x^2 − 2x + 2)
auto mix_eig = rational_eigenvalues(mix).value();
mix_eig.size() == 1 && mix_eig[0].first == bi(1);  // only the rational 1
total_mult(mix_eig) == 1;      // 0 < 1 < n = 3: partly rational

// --- Large entries: overflows int64, stays exact in the big tier -----------
// trace 8, det 15 (eigenvalues 3 and 5), but built with an entry near 10^10 and
// an off-diagonal near −10^20. The Faddeev–LeVerrier intermediates (~10^20)
// blow past the int64 ceiling, yet BigRational keeps everything exact.
auto big = BigMatrix::from_rows(
    {{brs("10000000000"), brs("-99999999920000000015")},
     {bi(1),              brs("-9999999992")}}).value();
characteristic_polynomial(big).value();      // {15, −8, 1}  (x^2 − 8x + 15)
auto big_eig = rational_eigenvalues(big).value();
big_eig[0].first == bi(3) && big_eig[1].first == bi(5);  // exact {3, 5}
total_mult(big_eig) == 2;      // fully rational spectrum recovered

// --- Determinant and inverse byproducts ------------------------------------
// The FL determinant agrees with BigMatrix's fraction-free Bareiss determinant.
auto a = bmat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
determinant(a).value() == a.determinant().value();  // true (both −306)
determinant(BigMatrix::identity(0)).value() == bi(1);  // 0×0 det = 1

// Exact fractional inverse: [[1,2],[3,4]]^{-1} = [[−2, 1], [3/2, −1/2]].
auto inv = inverse(bmat({{1, 2}, {3, 4}})).value();
bmat({{1, 2}, {3, 4}}).multiply(inv).value() == BigMatrix::identity(2);  // A·A⁻¹ = I
inv.at(1, 0) == brat(3, 2) && inv.at(1, 1) == brat(-1, 2);  // exact fractions

// A singular matrix has no inverse → domain_error.
inverse(bmat({{1, 2}, {2, 4}})).error();     // MathError::domain_error

// Non-square input is a domain error for every entry point.
auto wide = bmat({{1, 2, 3}, {4, 5, 6}});
characteristic_polynomial(wide).error();     // MathError::domain_error
rational_eigenvalues(wide).error();          // MathError::domain_error
```

## See also

- [`nimblecas.bigmatrix`](bigmatrix.md) — the unbounded `BigRational` matrix
  whose spectrum this module analyses (the input type).
- [`nimblecas.eigen`](eigen.md) — the `int64`-`Rational`, overflow-checked
  sibling this module big-backs; use it when values fit `int64`.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction field
  the coefficients and eigenvalues live in.
- [`nimblecas.bigint`](bigint.md) — the arbitrary-precision integer tier the
  rational-root search (denominator clearing, divisor enumeration) runs over.
- [`nimblecas.matrix`](matrix.md) — the `int64`-`Rational` matrix `bigmatrix`
  promotes from.
- [`nimblecas.roots`](roots.md) — the `int64` rational-root theorem solver that
  `eigen` reuses and this module reimplements over `BigInt` to avoid overflow.
- [Documentation hub](../Index.md)
