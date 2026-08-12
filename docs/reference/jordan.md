# `nimblecas.jordan` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/jordan/jordan.cppm`

The **Jordan canonical form** `J` of a square matrix `A` **together with the
transforming matrix `P`** such that

```
A = P * J * P^{-1}        (equivalently  A * P == P * J,  P invertible)
```

`J` is block-diagonal, one **Jordan block** per Jordan chain:

```
        [ l 1 0 ... 0 ]
        [ 0 l 1 ... 0 ]
   J_k =[ ...         ]     (eigenvalue l on the diagonal, 1 on the superdiagonal)
        [ 0 0 0 ... 1 ]
        [ 0 0 0 ... l ]
```

and the columns of `P` are the generalized eigenvectors of `A`, grouped into
chains and ordered so each chain fills one block.

```cpp
import nimblecas.jordan;
```

Depends on [`core`](core.md), [`matrix`](matrix.md), [`ratpoly`](ratpoly.md),
[`roots`](roots.md), [`eigen`](eigen.md), [`factor`](factor.md),
[`algnum`](algnum.md) and [`splitfield`](splitfield.md) (the Tier-3 general
splitting field), plus `bigint`, [`bigrational`](bigrational.md),
[`bigratpoly`](bigratpoly.md), [`bigalgnum`](bigalgnum.md), and
[`bigsplitfield`](bigsplitfield.md) (the Tier-3 bignum splitting field
`jordan_form_bignum` builds on).

## Honesty boundary — exact, extension, or honest refusal

Unlike the **Frobenius / rational canonical form** ([`frobenius`](frobenius.md)),
which is **always exact over `Q`** because it never needs the eigenvalues, the
Jordan form is built **from the eigenvalues**, so it can only be exact where the
eigenvalues can be represented exactly. This module is delivered in three honest
tiers (Rule 32 throughout — an **exact, verified** result or an honest
`MathError`; **never a wrong `P` and never a decimalized eigenvalue**):

| Tier | Function | When | Result |
| :--- | :--- | :--- | :--- |
| **1** | `rational_jordan_form` | char poly **splits over `Q`** (all eigenvalues rational) | `J`, `P` exact over `Q` |
| **2** | `jordan_form` | the only non-linear irreducible factor is a **single quadratic** `q(x)` (possibly repeated) | `J`, `P` exact over `Q(alpha) = Q[x]/(q)` |
| **3** | `jordan_form` / `jordan_form(A, max_field_degree)` | an irreducible factor of **degree ≥ 3**, or **two or more distinct** quadratic factors | `J`, `P` exact over the **general splitting field** built by [`splitfield`](splitfield.md) — capped at `max_field_degree` (default 12); honest `not_implemented` / `overflow` when that arithmetic envelope is exceeded |

For **every** returned `(J, P)`, the module **verifies exactly** that
`A * P == P * J` and that `P` is invertible (its kernel is trivial) *before*
returning. If that certificate cannot be produced, an error is returned instead
of an unverified `P`.

### Contrast with the Frobenius (rational canonical) form

`[[0,-1],[1,0]]` has characteristic polynomial `x^2 + 1` with eigenvalues `±i`.
The **rational canonical form is exact over `Q`** — its single invariant factor
is `x^2 + 1` and `RCF = [[0,-1],[1,0]]` — because it never names the eigenvalues.
The **Jordan form of the same matrix is `diag(i, -i)`**, which *does* name them,
so it is only exact once we move into `Q(i)`. That is precisely why `jordan_form`
constructs an extension field and `rational_canonical_form` never has to: the
finer, eigenvalue-resolved Jordan form pays for its resolution with a field
extension.

## Tier 1 — over `Q`

`rational_jordan_form(A)` applies when the characteristic polynomial splits over
`Q` (the rational-root multiplicities sum to `n`). For each eigenvalue `l`, with
`N = A − l*I`:

- the generalized eigenspace is `ker(N^k)` for the smallest `k` (the *index*)
  at which the nullity stabilises; kernels are computed by exact `Q`-RREF;
- **Jordan chains** are extracted by the standard top-down construction: for
  block size `k` from the index down to `1`, the vectors of `ker(N^k)` that are
  independent modulo `ker(N^{k-1})` **and** modulo the images carried down from
  longer chains are exactly the generators of the length-`k` chains. Applying
  `N` down a generator `g` gives the chain `g, N g, …, N^{k-1} g`, laid into `P`
  in eigenvector-first order so the block reads `l` on the diagonal, `1` on the
  superdiagonal.

`J` and `P` are returned as exact `Rational` `Matrix` values in
`RationalJordan{ jordan, transform }`.

A matrix whose characteristic polynomial does **not** split over `Q` is a
`domain_error` here — there is no Jordan form over `Q` — with `jordan_form` the
route for the single-quadratic-extension case.

## Tier 2 — over a quadratic extension `Q(alpha)`

`jordan_form(A)` factors the characteristic polynomial over `Q` with
[`factor_over_Q`](factor.md). When its only non-linear irreducible factor is a
single quadratic `q(x) = x^2 + b x + c` (with any multiplicity), that quadratic
**always splits** in `Q(alpha) = Q[x]/(q)`: `alpha` is one root and `−b − alpha`
is its conjugate (the two roots sum to `−b`), both elements of the field. Every
eigenvalue — the rational ones (embedded via `from_rational`) and the conjugate
pair — is represented as an [`AlgebraicNumber`](algnum.md), and the **entire**
generalized-eigenvector computation (RREF, null spaces, chain extraction) is done
exactly over `Q(alpha)` using its `is_zero` / `inverse` / `multiply` / `subtract`
arithmetic.

The result `AlgebraicJordan{ field, jordan, transform }` carries the
`NumberField` together with `J` and `P` as dense row-major
`std::vector<std::vector<AlgebraicNumber>>` matrices over that field. Every entry
is an exact residue in `Q(alpha)` — never a floating-point stand-in.

**Concrete cases handled:** `[[0,-1],[1,0]]` → `Q(i)`, `J = diag(i, −i)`;
`[[2,-1],[1,2]]` → `Q(alpha)` with `alpha^2 − 4 alpha + 5 = 0`, `J = diag(2+i, 2−i)`;
`[[C, I₂],[0, C]]` with `C = [[0,-1],[1,0]]` → char poly `(x^2+1)^2`, a **defective**
repeated complex pair whose `J` has one size-2 Jordan block for each of `i` and `−i`.

## Tier 3 — the general splitting field, bounded by an exact-arithmetic envelope

When the factorization of the characteristic polynomial over `Q` does **not** fit
the single-quadratic Tier 2 case — because it contains either

1. an irreducible factor of **degree ≥ 3**, or
2. **two or more distinct** irreducible quadratic factors —

`jordan_form` does **not** simply refuse. It builds the **one common splitting
field** of *every* non-linear factor at once via
[`splitfield`](splitfield.md)`::splitting_field`, rebuilds every eigenvalue (the
rational ones and every harvested root of every non-linear factor) as an
[`AlgebraicNumber`](algnum.md) in that single field, and runs the **same**
`compute_groups → assemble → verify` pipeline as Tier 2 — so a returned `(J, P)`
is verified exactly (`A * P == P * J`, `P` invertible) over the extension, just as
in Tiers 1 and 2.

The splitting-field construction is **capped at `max_field_degree`**
(`jordan_form(A)` uses `kDefaultMaxSplittingFieldDegree = 12`; the
`jordan_form(A, max_field_degree)` overload lets the caller raise or lower it).
The cap exists because the splitting field's degree can grow fast — a quartic with
Galois group `S₄` needs degree 24 — and because the field is built with **exact
`int64`-backed `Rational` arithmetic**: Trager's norm/resultant construction and
the primitive-element search inflate coefficients quickly, so even fields whose
*degree* is within the cap (e.g. the degree-6 splitting field of `x³ − 2`) can
exhaust the `int64` numerator/denominator range.

When either bound is hit, `jordan_form` returns an **honest** error —
`MathError::not_implemented` when `splitting_field`'s degree/budget envelope is
exceeded, or `MathError::overflow` on an `int64` overflow in the exact arithmetic
— propagated verbatim from `splitfield`, never fabricated. In every such case
**`jordan_structure` (Tier 0) remains the exact-over-`Q` fallback**: it needs no
splitting field and so is unaffected by this envelope. No wrong, unverified, or
decimalized `(J, P)` is ever produced — the guarantee is identical across all
three tiers; only the *reach* of Tier 3 is bounded by the `int64` substrate.

> **Substrate note.** This bound is a property of the `int64`-backed `Rational`
> arithmetic the extension is built on, **not** of the algorithm: the same
> `splitting_field → compute_groups → assemble → verify` path over a bignum-backed
> rational extends Tier 3's reach without any change to `jordan_form`'s logic — this is
> exactly what `jordan_form_bignum` below **realizes**.

## Tier 3 (bignum) — the general splitting field, over the UNBOUNDED rationals

`jordan_form_bignum(A)` / `jordan_form_bignum(A, max_field_degree)` is the
**arbitrary-precision mirror** of `jordan_form`'s general splitting-field path
(Tier 3 above). It builds the splitting field of the characteristic
polynomial's non-linear irreducible factors on `BigRational` via
[`nimblecas.bigsplitfield`](bigsplitfield.md) instead of the `int64`
[`nimblecas.splitfield`](splitfield.md), so it **never** fails with
`MathError::overflow` from the splitting-field construction — that is exactly
the ceiling this tier removes.

```cpp
struct BigAlgebraicJordan {
    BigNumberField field;
    std::vector<std::vector<BigAlgebraicNumber>> jordan;     // J, n x n over `field`
    std::vector<std::vector<BigAlgebraicNumber>> transform;  // P, n x n over `field`
};

[[nodiscard]] auto jordan_form_bignum(const Matrix& a) -> Result<BigAlgebraicJordan>;
[[nodiscard]] auto jordan_form_bignum(const Matrix& a, std::int64_t max_field_degree)
    -> Result<BigAlgebraicJordan>;
```

The single-argument overload uses `kDefaultMaxSplittingFieldDegree` (12), same
as `jordan_form`. The **characteristic polynomial and its `Q`-factorization
are still computed on the `int64` tier** — their coefficients are small,
bounded by `A`'s own entries, so overflow does not strike there. Overflow only
ever struck *later*, inside the splitting-field arithmetic (Trager norms,
primitive-element multiplication matrices), which is precisely the step this
tier moves onto `BigRational`. Only the non-linear irreducible factors are
lifted to `BigRationalPoly` for `bigsplitfield::splitting_field`; `A` and the
rational eigenvalues are then embedded into the one common field, and the
**same** `compute_groups → assemble → verify` pipeline that Tiers 2/3 use runs
again, this time instantiated with `S = BigAlgebraicNumber`.

**Headline.** The companion matrix of `x^3 − 2` — which the `int64`
`jordan_form` must refuse (its degree-6 splitting field `Q(2^{1/3}, ω)`
overflows `int64`, see the Tier 3 example above) — **diagonalizes exactly**
under `jordan_form_bignum`: `J = diag(r1, r2, r3)` with each `ri^3 == 2`,
`A*P == P*J` verified over `Q(2^{1/3}, ω)` (see the
`cubic_splitting_field_bignum_diagonalizes` test).

Fails with:

- `domain_error` — `A` is not square or is `0x0`, **or** the characteristic
  polynomial splits over `Q` (no extension is needed — use
  `rational_jordan_form` instead);
- `not_implemented` — the bignum splitting field would exceed
  `max_field_degree` at some point, or one of `bigsplitfield`'s own internal
  budgets (Trager's shift search, the primitive-element search,
  `factor_over_Q`'s budget) is exceeded;
- `overflow` — **only** from the `int64` `characteristic_polynomial` /
  `factor_over_Q` pre-pass, **never** from the splitting-field construction
  itself.

## Tier 0 — `jordan_structure`: the Segre characteristic over `Q` alone, no extension field

`jordan_structure(A)` is a **fourth, independent** answer: the exact-over-`Q`
Jordan block **structure** (the *Segre characteristic* — the multiset of
block sizes per eigenvalue) of **any** square rational matrix, including one
whose eigenvalues are irrational or complex — **without constructing the
splitting field they live in**. It complements `jordan_form` (Tiers 1–3
above), which additionally builds the transforming matrix `P` and therefore
*needs* the eigenvalues to be representable (rational, or in a single
quadratic extension); `jordan_structure` needs neither, because **rank is
invariant under field extension**.

For each irreducible factor `m_i(x)` of the characteristic polynomial (degree
`d_i`, multiplicity `e_i`), every one of `m_i`'s `d_i` conjugate roots has the
**same** Jordan block-size partition of `e_i` — by Galois symmetry: `A` is
rational, so `m_i`'s conjugate roots are indistinguishable to any rational
invariant such as rank. `jordan_structure` therefore returns one shared
partition per irreducible factor rather than one per numbered eigenvalue:

```cpp
struct JordanFactorStructure {
    RationalPoly factor;                    // the irreducible factor m_i(x)
    std::int64_t degree;                     // d_i = deg(m_i) -- number of conjugate roots
    std::int64_t multiplicity;               // e_i -- multiplicity in the characteristic poly
    std::vector<std::int64_t> block_sizes;   // descending; sum == multiplicity
};
struct JordanStructure {
    std::vector<JordanFactorStructure> factors;  // one per distinct irreducible factor,
                                                   // canonical order: degree ascending, then
                                                   // coefficient-lexicographic among ties
};
```

**Algorithm.** With `p` the characteristic polynomial, factored over `Q` into
`(m_i, e_i)` pairs: for each factor of degree `d_i`, with `M = m_i(A)`
(Horner-evaluated in the `Matrix` ring), `nu_k = (n − rank(M^k)) / d_i` is the
total generalized-eigenspace dimension at level `k`, shared out evenly over
the `d_i` conjugate roots; the standard nullity identity
`#blocks_of_size_k = 2·nu_k − nu_{k−1} − nu_{k+1}` (`nu_0 = 0`) then recovers
the Segre characteristic common to every conjugate root of `m_i`.

**Rule-32 divisibility guard.** `(n − rank(M^k))` must divide evenly by `d_i`
at every step, and the recovered block-size partition must sum exactly to
`e_i`; either check failing returns `MathError::domain_error` rather than
truncate-dividing to a plausible-looking but wrong Segre characteristic. Both
are unreachable for a correct characteristic polynomial and exact arithmetic
— they exist purely as an honesty tripwire, not an expected code path.

`jordan_structure` answers cases `jordan_form` cannot: e.g. the defective
repeated-cubic matrix `A = [[C, I₃], [0, C]]` with `C = companion(x³ − 3x − 1)`
has characteristic polynomial `(x³ − 3x − 1)²` — a single degree-3 irreducible
factor at multiplicity 2, which is Tier-3 `not_implemented` for `jordan_form`
(degree ≥ 3) — yet `jordan_structure` computes its exact Segre characteristic
(`block_sizes == {2}`: each of the 3 conjugate roots carries one defective
2×2 block) entirely over `Q`.

| Function / type | Signature | Behavior |
| :--- | :--- | :--- |
| `jordan_structure` | `auto jordan_structure(const Matrix& a) -> Result<JordanStructure>` | The exact-over-`Q` Jordan block structure of `A` — valid for any square rational matrix, irrational/complex eigenvalues included, without building a splitting field. The `0×0` matrix yields the empty structure. |

## API

| Function / type | Signature | Behavior |
| :--- | :--- | :--- |
| `RationalJordan` | `struct { Matrix jordan; Matrix transform; }` | Tier 1 result: exact `Q` matrices `J` and `P` with `A*P == P*J`, `P` invertible. |
| `rational_jordan_form` | `auto rational_jordan_form(const Matrix& a) -> Result<RationalJordan>` | Tier 1. Char poly splits over `Q`. The 0×0 matrix yields empty `J`, `P`. Non-square **or** non-splitting ⇒ `domain_error`; overflow propagated. |
| `AlgebraicJordan` | `struct { NumberField field; vector<vector<AlgebraicNumber>> jordan; vector<vector<AlgebraicNumber>> transform; }` | Tier 2 result: `J` and `P` over `field = Q(alpha)`, with `A*P == P*J` over the field, `P` invertible. |
| `jordan_form` | `auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan>` | Tier 2/3. Single quadratic extension (Tier 2), else the general splitting field capped at `kDefaultMaxSplittingFieldDegree = 12` (Tier 3). Char poly splitting over `Q` (or 0×0, or non-square) ⇒ `domain_error`; splitting-field envelope exceeded ⇒ `not_implemented`; `int64` overflow propagated. |
| `jordan_form` (capped) | `auto jordan_form(const Matrix& a, std::int64_t max_field_degree) -> Result<AlgebraicJordan>` | As above, with a caller-chosen Tier-3 splitting-field degree cap in place of the default 12. |
| `BigAlgebraicJordan` | `struct { BigNumberField field; vector<vector<BigAlgebraicNumber>> jordan; vector<vector<BigAlgebraicNumber>> transform; }` | Tier 3 (bignum) result: `J` and `P` over the **unbounded** `field = BigNumberField`, with `A*P == P*J` over the field, `P` invertible. |
| `jordan_form_bignum` | `auto jordan_form_bignum(const Matrix& a) -> Result<BigAlgebraicJordan>` | Tier 3, over the unbounded splitting field built by [`bigsplitfield`](bigsplitfield.md) — **never** overflows building the field itself (e.g. the companion matrix of `x^3 − 2`, which the `int64` `jordan_form` must refuse). Char poly splitting over `Q` (or 0×0, or non-square) ⇒ `domain_error`; bignum splitting-field envelope exceeded ⇒ `not_implemented`; `overflow` only from the `int64` characteristic-polynomial/`factor_over_Q` pre-pass. |
| `jordan_form_bignum` (capped) | `auto jordan_form_bignum(const Matrix& a, std::int64_t max_field_degree) -> Result<BigAlgebraicJordan>` | As above, with a caller-chosen Tier-3 splitting-field degree cap in place of the default 12. |
| `JordanFactorStructure` / `JordanStructure` / `jordan_structure` | see above | Tier 0. Exact-over-`Q` Segre characteristic for **any** square rational matrix; no extension field ever built. |

## Error model

| Condition | Error |
| :--- | :--- |
| Non-square input to any of the three functions | `MathError::domain_error` |
| `rational_jordan_form`: characteristic polynomial does not split over `Q` | `MathError::domain_error` |
| `jordan_form`: characteristic polynomial splits over `Q` (no extension needed) | `MathError::domain_error` |
| `jordan_form`: 0×0 input (no eigenvalues, no field) | `MathError::domain_error` |
| `jordan_form`: Tier-3 splitting field would exceed `max_field_degree`, or any `splitfield` internal budget (Trager shift search, primitive-element search) is exhausted | `MathError::not_implemented` (propagated from `splitfield`) |
| `jordan_form_bignum`: 0×0 input, non-square input, or characteristic polynomial splits over `Q` (no extension needed) | `MathError::domain_error` |
| `jordan_form_bignum`: bignum splitting field would exceed `max_field_degree`, or any `bigsplitfield` internal budget (Trager shift search, primitive-element search, `factor_over_Q` budget) is exhausted | `MathError::not_implemented` (propagated from `bigsplitfield`) |
| `jordan_form_bignum`: `int64` overflow in the `characteristic_polynomial` / `factor_over_Q` pre-pass (never in the splitting-field construction itself) | `MathError::overflow` |
| `jordan_structure`: the divisibility or partition-sum Rule-32 guard trips (unreachable for correct exact arithmetic) | `MathError::domain_error` |
| `jordan_structure`: propagated `characteristic_polynomial` / `factor_over_Q` errors (e.g. `factor_over_Q`'s Kronecker search exceeding its divisor-tuple budget) | `MathError::not_implemented` (or whatever is propagated) |
| `int64` numerator/denominator overflow in the exact arithmetic (any of the three functions) | `MathError::overflow` |

## Examples

```cpp
import nimblecas.jordan;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};
// Low-degree-first integer coefficients, e.g. poly({1, 0, 1}) == x^2 + 1.
auto poly = [](std::vector<std::int64_t> c) {
    std::vector<Rational> rc;
    for (auto v : c) rc.push_back(ri(v));
    return RationalPoly::from_coeffs(std::move(rc));
};

// Tier 1 — a defective 2x2 block over Q.
auto A = mat({{ri(5), ri(1)}, {ri(-1), ri(3)}});    // char poly (x-4)^2
auto j = rational_jordan_form(A).value();
// j.jordan     == [[4,1],[0,4]]
// j.transform  == [[1,0],[-1,1]]      and A * j.transform == j.transform * j.jordan

// Tier 2 — a rotation, exact over Q(i).
auto R = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});    // char poly x^2 + 1
auto k = jordan_form(R).value();
// k.field      == Q[x]/(x^2 + 1)
// k.jordan     == diag(i, -i),  k.transform == [[i,-i],[1,1]]   over Q(i)

// Tier 3 — the general splitting-field path, honestly bounded by the int64 substrate.
auto C = mat({{ri(0), ri(0), ri(2)},                // companion of x^3 - 2
              {ri(1), ri(0), ri(0)},
              {ri(0), ri(1), ri(0)}});
auto e = jordan_form(C);   // x^3-2 splits in the degree-6 field Q(2^{1/3}, w); that degree
                           // is within the default cap of 12, but building it in int64
                           // Rational arithmetic overflows -> honest not_implemented / overflow.
                           // jordan_structure(C) still returns the exact Segre data over Q.

// Tier 3 (bignum) — the SAME companion matrix C, exact this time: no int64 ceiling.
auto f = jordan_form_bignum(C).value();
// f.field.degree() == 6                             -- Q(2^{1/3}, w), built on BigRational
// f.jordan == diag(r1, r2, r3), each ri^3 == 2       -- three simple, pairwise-distinct roots
// A * f.transform == f.transform * f.jordan          -- verified exactly over the bignum field

// Tier 0 — jordan_structure answers the SAME rotation Tier 2 needed Q(i) for,
// entirely over Q: char poly x^2+1 is irreducible, one degree-2 factor,
// multiplicity 1, and (i, -i) are each simple => block_sizes {1}. No field built.
auto s = jordan_structure(R).value();
// s.factors.size() == 1
// s.factors[0].factor.is_equal(poly({1, 0, 1}))   -- x^2 + 1
// s.factors[0].degree == 2, s.factors[0].multiplicity == 1
// s.factors[0].block_sizes == std::vector<std::int64_t>{1}
```

## See also

- [`nimblecas.frobenius`](frobenius.md) — the rational canonical form: always
  exact over `Q`, but it returns only the canonical form (no `P`) because it
  never resolves the eigenvalues that Jordan needs.
- [`nimblecas.algnum`](algnum.md) — the `Q(alpha) = Q[x]/(m)` extension field and
  its `AlgebraicNumber` arithmetic that the Tier 2 linear algebra runs on.
- [`nimblecas.eigen`](eigen.md) — the characteristic polynomial (reused here) and
  the exact rational eigenvalues.
- [`nimblecas.factor`](factor.md) — factorization over `Q`, used to detect the
  quadratic factor and route the high-degree / multiple-quadratic cases to Tier 3.
- [`nimblecas.splitfield`](splitfield.md) — the general splitting-field construction
  (Trager factoring, root adjunction) that powers Tier 3, and whose `int64`-Rational
  envelope bounds Tier 3's reach.
- [`nimblecas.bigsplitfield`](bigsplitfield.md) — the unbounded, overflow-free mirror
  of `splitfield` that `jordan_form_bignum` (Tier 3, bignum) builds its splitting
  field on; removes Tier 3's `int64` ceiling entirely.
- [`nimblecas.matrix`](matrix.md) / [`nimblecas.ratpoly`](ratpoly.md) — the exact
  `Rational` matrix and `Q[x]` polynomial substrate.
- [Documentation hub](../Index.md)
```
