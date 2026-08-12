# `nimblecas.bigsplitfield` — Splitting-field construction over Q, unbounded (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigsplitfield/bigsplitfield.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.splitfield`](splitfield.md).
Where that module carries `int64` `Rational` / `RationalPoly` coefficients and
surfaces `MathError::overflow` the moment an intermediate numerator or
denominator saturates, this one carries `BigRational` / `BigRationalPoly`
coefficients — via [`BigNumberField`](bigalgnum.md) / `BigAlgebraicNumber` /
[`BigAlgebraicPoly`](bigalgpoly.md) — and has **no such ceiling**:
`MathError::overflow` can **never** be produced. It builds splitting fields
over **Q** by Trager's norm-based factorization plus primitive-element
adjunction, exactly as `splitfield` does, but on the exact unbounded
substrate.

**The headline capability.** This module builds the **degree-6 splitting
field of `x^3 − 2`** — `Q(2^{1/3}, ω)` — **exactly**: the case whose Trager
norms and primitive-element multiplication matrices overflow the `int64`
`splitfield` tier before it can even attempt Kronecker factorization.

```cpp
import nimblecas.bigsplitfield;
```

Depends on [`core`](core.md), [`bigratpoly`](bigratpoly.md),
[`bigrational`](bigrational.md), [`bigmatrix`](bigmatrix.md),
[`bigalgnum`](bigalgnum.md), [`bigalgpoly`](bigalgpoly.md),
[`bigfactor`](bigfactor.md), [`bigresultant`](bigresultant.md), and
[`bigfrobenius`](bigfrobenius.md).

## The honesty boundary

- **`MathError::overflow` is never produced.** Every arithmetic step —
  Trager's evaluation-interpolation norm, the primitive-element
  multiplication-matrix build, the linear solves — runs over `BigRational` /
  `BigRationalPoly` / `BigMatrix`, none of which can overflow.
- **The critical divergence from the `int64` `splitfield`.** There,
  `adjoin_root` leans on `NumberField::create` to **reject** a reducible
  candidate minimal polynomial — that rejection *is* its irreducibility
  signal. `BigNumberField::create` does **not** prove irreducibility (this
  tier has no bignum polynomial-factoring layer built into number-field
  construction itself). So `adjoin_root` here additionally verifies the
  candidate primitive element's minimal polynomial is **Q-irreducible** via
  [`bigfactor::factor_over_Q`](bigfactor.md) — a single irreducible factor of
  multiplicity 1 and degree exactly `n = [L:Q]*deg(g)`. A nontrivial
  factorization means the candidate was **not** primitive (the bounded search
  continues over the next candidate); exhausting the search (bound 64) is
  `MathError::not_implemented`. This preserves the exact honesty invariant
  (Rule 32) that the `int64` tier gets for free from `NumberField::create`.
- **The driver (`splitting_field`) uses incremental root-division.** Rather
  than re-factor each *full* input polynomial over every enlarged field, it
  keeps, per input, only the **not-yet-split cofactor**, and divides a root
  out of it the moment that root is found. Every polynomial ever handed to
  `factor_over_field` is therefore the reduced cofactor — for `x^3 − 2` its
  Trager norm degree never exceeds `[L:Q]*deg(cofactor) <= 6`, never the
  degree-18 norm that re-factoring the full cubic over the finished degree-6
  field would demand (well beyond `factor_over_Q`'s Kronecker budget). This
  is precisely what makes the headline case tractable. Every result is
  re-verified before returning: each irreducible factor contributes exactly
  `deg_i` pairwise-distinct roots, and the reconstruction
  `f == lc(f) * prod(f_i)` is checked in `factor_over_field` itself.
- **`MathError::not_implemented`** — `max_degree` exceeded, at the initial
  field or any subsequent adjunction; Trager's shift search exhausted (no
  shift yields a squarefree norm within `|s| <= 2*[L:Q]*deg(f)`); the
  primitive-element search in `adjoin_root` exhausted (bound 64); or
  `factor_over_Q`'s own internal budget (trial-division / divisor-tuple)
  propagated verbatim.
- **`MathError::domain_error`** — `f` / `g` does not live in the field `l`
  passed to it; `g`'s degree `< 2` in `adjoin_root`; a non-positive-degree
  entry in `splitting_field`'s input; or any post-construction
  embedding-verification / completeness-guard failure (unreachable for
  correct exact arithmetic — an honesty tripwire, not an expected path).

## API summary

```cpp
// Factor a SQUAREFREE f in L[x] into its monic L-irreducible factors, via Trager's norm
// algorithm. f must live in `l` (else domain_error); the zero polynomial is domain_error.
// A nonzero constant f yields an empty list (no non-unit factors); a linear f yields itself
// (monic). Fails with MathError::not_implemented when no shift in the bounded search
// |s| <= 2*[L:Q]*deg(f) yields a squarefree norm, or when factor_over_Q(N) exceeds its own
// budget. The exact bignum arithmetic never overflows. The reconstruction
// f == lc(f) * prod(factors) is verified before returning (domain_error on mismatch).
[[nodiscard]] auto factor_over_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicPoly>>;

// The roots of f that lie in L, read off the degree-1 factors of factor_over_field(l, f)
// as lambda = -c0/c1. Order matches factor_over_field's (unspecified but deterministic)
// factor order. Errors propagate from factor_over_field.
[[nodiscard]] auto roots_in_field(const BigNumberField& l, const BigAlgebraicPoly& f)
    -> Result<std::vector<BigAlgebraicNumber>>;

// A simple extension M >= L containing a root of g, plus the images in M of L's generator
// (`old_generator`, needed to re-embed any other element of L into M) and of the adjoined
// root itself (`root`).
struct BigAdjoinedRoot {
    BigNumberField field;
    BigAlgebraicNumber old_generator;
    BigAlgebraicNumber root;
};

// Adjoin a root of g (irreducible over L, deg g = e >= 2) to L, building M = Q(gamma') as a
// SIMPLE extension via a primitive-element search gamma' = beta + c*alpha over
// c = 1, -1, 2, -2, .... Fails with domain_error when g does not live in `l` or has degree
// < 2; with MathError::not_implemented when no primitive element is found within the bounded
// search (this is also the honest signal if g was not actually irreducible over L, since this
// tier proves primitivity by verifying h' is Q-irreducible via factor_over_Q -- see the module
// header), or when factor_over_Q exceeds its own budget; with domain_error if the mandatory
// post-construction verification (g(root) == 0 and L.modulus()(old_generator) == 0, both
// checked in M) fails. The exact bignum arithmetic never overflows.
[[nodiscard]] auto adjoin_root(const BigNumberField& l, const BigAlgebraicPoly& g)
    -> Result<BigAdjoinedRoot>;

// The splitting field of a batch of Q-irreducible polynomials, together with each
// polynomial's full set of roots in that field (in the SAME order as `irreducibles`).
struct BigSplittingField {
    BigNumberField field;
    std::vector<std::pair<BigRationalPoly, std::vector<BigAlgebraicNumber>>> roots;
};

// Build the splitting field of `irreducibles` (each assumed Q-irreducible; a non-positive
// degree entry is domain_error), refusing to build a field of degree exceeding
// `max_degree` at any point (the initial field or any subsequent adjunction) with
// MathError::not_implemented -- the honest boundary, since e.g. a quartic with Galois
// group S4 needs degree 24. Errors from factor_over_field / adjoin_root propagate
// (not_implemented / domain_error; overflow is impossible on this tier). A defensive
// completeness guard (every factor contributes exactly deg_i pairwise-distinct roots) is
// re-verified before returning; a violation is domain_error (an embedding bug, not a
// mathematical possibility for a genuinely Q-irreducible, hence separable, input).
[[nodiscard]] auto splitting_field(std::span<const BigRationalPoly> irreducibles,
                                    std::int64_t max_degree) -> Result<BigSplittingField>;
```

## Error model

| Condition | Error |
| :--- | :--- |
| `factor_over_field`: `f` does not live in `l`, or `f` is the zero polynomial | `MathError::domain_error` |
| `factor_over_field`: Trager's shift search exhausted (no squarefree norm), or `factor_over_Q(N)` exceeds its own budget | `MathError::not_implemented` |
| `factor_over_field`: the reconstructed product `lc(f) * prod(factors)` fails to equal `f` | `MathError::domain_error` (unreachable for correct arithmetic) |
| `adjoin_root`: `g` does not live in `l`, or `deg(g) < 2` | `MathError::domain_error` |
| `adjoin_root`: primitive-element search exhausted (bound 64), or `factor_over_Q` exceeds its own budget | `MathError::not_implemented` |
| `adjoin_root`: post-construction verification (`g(root) == 0`, `L.modulus()(old_generator) == 0`) fails | `MathError::domain_error` (unreachable for correct arithmetic) |
| `splitting_field`: a non-positive-degree input polynomial | `MathError::domain_error` |
| `splitting_field`: the field degree would exceed `max_degree` at any point (initial field or a subsequent adjunction) | `MathError::not_implemented` |
| `splitting_field`: the completeness guard (each factor contributes exactly `deg_i` pairwise-distinct roots) fails | `MathError::domain_error` (unreachable for correct arithmetic) |
| Errors propagated from `factor_over_field` / `adjoin_root` | as above |

There is **no `overflow` row**: `BigRational` / `BigRationalPoly` /
`BigMatrix` arithmetic cannot wrap, so the `overflow` the `int64`
[`splitfield`](splitfield.md) can raise is unreachable here.

## Worked example

```cpp
import nimblecas.bigsplitfield;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
import nimblecas.bigalgnum;
using namespace nimblecas;

auto q = [](std::int64_t v) { return BigRational::from_int(v); };
auto poly = [](std::initializer_list<std::int64_t> cs) {
    std::vector<BigRational> v;
    for (std::int64_t c : cs) v.push_back(q(c));
    return BigRationalPoly::from_coeffs(std::move(v));
};

// HEADLINE: the degree-6 splitting field of x^3 - 2, Q(2^{1/3}, w) -- the case the int64
// splitfield overflows on. Three harvested roots, each cubing to 2, pairwise distinct.
const std::vector<BigRationalPoly> in{poly({-2, 0, 0, 1})};   // x^3 - 2
auto sf = splitting_field(in, 6).value();
sf.field.degree() == 6;                                       // [Q(2^{1/3}, w) : Q] == 6
const auto& roots = sf.roots[0].second;
roots.size() == 3;                                             // three cube roots of 2
const BigAlgebraicNumber two = sf.field.from_bigrational(q(2));
roots[0].pow(3).value().is_equal(two);                         // r^3 == 2, exactly
!roots[0].is_equal(roots[1]) && !roots[0].is_equal(roots[2]) &&
    !roots[1].is_equal(roots[2]);                               // pairwise distinct

// factor_over_field(Q(i), x^2 + 1) == (x - i)(x + i).
auto qi = BigNumberField::create(poly({1, 0, 1})).value();     // x^2 + 1
const BigAlgebraicPoly embedded = BigAlgebraicPoly::embed(qi, poly({1, 0, 1}));
auto factors = factor_over_field(qi, embedded).value();
factors.size() == 2 && factors[0].degree() == 1 && factors[1].degree() == 1;
auto roots_qi = roots_in_field(qi, embedded).value();
roots_qi[0].negate().value().is_equal(roots_qi[1]);            // the roots are i and -i

// The honest refusal: x^3 - 2 needs degree 6, but max_degree = 2 is too small.
splitting_field(in, 2).error() == MathError::not_implemented;
```

## See also

- [`nimblecas.splitfield`](splitfield.md) — the `int64`-`Rational`,
  overflow-checked sibling this module big-backs; use it when values fit
  `int64`.
- [`nimblecas.bigalgnum`](bigalgnum.md) — the `BigNumberField` /
  `BigAlgebraicNumber` unbounded extension-field substrate.
- [`nimblecas.bigalgpoly`](bigalgpoly.md) — the `BigAlgebraicPoly`
  dense-polynomial-over-a-field substrate Trager's algorithm runs on.
- [`nimblecas.bigfactor`](bigfactor.md) — `factor_over_Q`, used both to
  factor Trager norms and to prove primitive-element irreducibility.
- [`nimblecas.bigresultant`](bigresultant.md) — the field-norm resultant
  substrate Trager's algorithm is built on.
- [`nimblecas.bigfrobenius`](bigfrobenius.md) — the minimal-polynomial
  computation `adjoin_root` uses to test primitivity.
- [`nimblecas.jordan`](jordan.md) — the Tier-3 bignum consumer
  (`jordan_form_bignum`) that builds its splitting field via this module.
- [Documentation hub](../Index.md)
