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
[`roots`](roots.md), [`eigen`](eigen.md), [`factor`](factor.md) and
[`algnum`](algnum.md).

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
| **3** | `jordan_form` | an irreducible factor of **degree ≥ 3**, or **two or more distinct** quadratic factors | `MathError::not_implemented` |

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

## Tier 3 — the precise refusal boundary

`jordan_form` returns `MathError::not_implemented` when, and only when, the exact
factorization of the characteristic polynomial over `Q` contains either

1. an irreducible factor of **degree ≥ 3** — a general splitting field, which
   would require an algebraic extension beyond a single quadratic; or
2. **two or more distinct** irreducible quadratic factors — even if they happen
   to share a splitting field, this module does not attempt to unify them into a
   common `Q(alpha)`, so it refuses rather than risk an unsupported or composite
   extension.

This boundary is deliberately conservative: everything inside it is computed and
**verified exactly**; everything outside it is refused honestly. No wrong or
decimalized answer is ever produced.

## API

| Function / type | Signature | Behavior |
| :--- | :--- | :--- |
| `RationalJordan` | `struct { Matrix jordan; Matrix transform; }` | Tier 1 result: exact `Q` matrices `J` and `P` with `A*P == P*J`, `P` invertible. |
| `rational_jordan_form` | `auto rational_jordan_form(const Matrix& a) -> Result<RationalJordan>` | Tier 1. Char poly splits over `Q`. The 0×0 matrix yields empty `J`, `P`. Non-square **or** non-splitting ⇒ `domain_error`; overflow propagated. |
| `AlgebraicJordan` | `struct { NumberField field; vector<vector<AlgebraicNumber>> jordan; vector<vector<AlgebraicNumber>> transform; }` | Tier 2 result: `J` and `P` over `field = Q(alpha)`, with `A*P == P*J` over the field, `P` invertible. |
| `jordan_form` | `auto jordan_form(const Matrix& a) -> Result<AlgebraicJordan>` | Tier 2/3. Single quadratic extension. Degree-≥3 factor or ≥2 distinct quadratics ⇒ `not_implemented`; char poly splitting over `Q` (or 0×0, or non-square) ⇒ `domain_error`; overflow propagated. |

## Error model

| Condition | Error |
| :--- | :--- |
| Non-square input to either function | `MathError::domain_error` |
| `rational_jordan_form`: characteristic polynomial does not split over `Q` | `MathError::domain_error` |
| `jordan_form`: characteristic polynomial splits over `Q` (no extension needed) | `MathError::domain_error` |
| `jordan_form`: 0×0 input (no eigenvalues, no field) | `MathError::domain_error` |
| `jordan_form`: irreducible factor of degree ≥ 3, or ≥ 2 distinct quadratic factors | `MathError::not_implemented` |
| `int64` numerator/denominator overflow in the exact arithmetic | `MathError::overflow` |

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

// Tier 3 — an honest refusal.
auto C = mat({{ri(0), ri(0), ri(2)},                // companion of x^3 - 2
              {ri(1), ri(0), ri(0)},
              {ri(0), ri(1), ri(0)}});
auto e = jordan_form(C);                             // error == MathError::not_implemented
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
  quadratic factor and the out-of-scope high-degree / multiple-quadratic cases.
- [`nimblecas.matrix`](matrix.md) / [`nimblecas.ratpoly`](ratpoly.md) — the exact
  `Rational` matrix and `Q[x]` polynomial substrate.
- [Documentation hub](../Index.md)
```
