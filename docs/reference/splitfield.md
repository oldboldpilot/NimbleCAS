# `nimblecas.splitfield` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/splitfield/splitfield.cppm`

Splitting-field construction over an algebraic number field: Trager's norm-based factorisation of dense polynomials
(`AlgebraicPoly`) over an algebraic number field (`NumberField`), adjunction of a single root of an irreducible
extension polynomial via a primitive-element search, and iterated adjunction to construct the splitting field of a
collection of $\mathbb{Q}$-irreducible polynomials. This module forms part of the algebra substrate (§7) and underpins
exact eigenvalue and Jordan canonical form calculations over algebraic extensions ([`nimblecas.jordan`](jordan.md)).

The module adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Exact rational arithmetic and decidability:** Every operation is exact over $\mathbb{Q}$. Field elements and
  polynomial coefficients are represented by exact `Rational` and `RationalPoly` values. There is no floating-point
  arithmetic, no numerical tolerance, and no heuristic approximation anywhere in this module. Nothing throws; every
  fallible operation returns `Result<T> = std::expected<T, MathError>`. If an exact result cannot be computed within the
  configured degree bound or arithmetic limits, the function returns an honest `MathError` rather than guessing or
  silently truncating.
- **Trager's norm algorithm (`factor_over_field`):** Given an algebraic number field $L = \mathbb{Q}(\gamma) =
  \mathbb{Q}[t]/(h(t))$ and a squarefree polynomial $f \in L[x]$, the algorithm selects an integer shift $s$ from the
  deterministic sequence $0, 1, -1, 2, -2, \dots$ and defines the shifted polynomial $g(x) = f(x - s\gamma)$. The field
  norm $N(x) = \operatorname{Norm}_{L/\mathbb{Q}}(g)$ is a polynomial in $\mathbb{Q}[x]$ of degree $[L:\mathbb{Q}] \cdot
  \deg(f)$. Because the field modulus $h$ is monic, the field norm $\operatorname{Norm}_{L/\mathbb{Q}}(b)$ for an element
  $b \in L$ (represented by a canonical residue polynomial in $t$ of degree $< [L:\mathbb{Q}]$) is given directly by the
  unnormalised resultant $\operatorname{Res}_t(h(t), b(t))$. The polynomial $N(x)$ is reconstructed via
  evaluation-interpolation across $\deg(N) + 1$ deterministic rational sample points. If $N(x)$ is not squarefree
  (detected when $\deg(\gcd(N, N')) > 0$), the shift was unlucky, and the search tests subsequent values of $s$ up to the
  bound $|s| \le 2[L:\mathbb{Q}]\deg(f)$. Once a squarefree norm is obtained, $N$ is factored over $\mathbb{Q}$ into
  irreducible components $N_i$ via `factor_over_Q(N)`. The corresponding $L$-factors of $g$ are computed via
  $\gcd_L(g, \operatorname{embed}_L(N_i))$, shifted back via $x \mapsto x + s\gamma$, and normalised to monic to yield
  the irreducible factors of $f$.
- **Factorisation reconstruction tripwire:** Before returning from `factor_over_field`, the recovered monic factors are
  multiplied together and scaled by $f$'s leading coefficient, verifying that $f == \operatorname{lc}(f) \prod f_i$
  identically in $L[x]$. A discrepancy indicates an internal representation or embedding bug, not a mathematical
  possibility, and is reported as `MathError::domain_error` rather than presented as fact.
- **Primitive-element search and irreducibility enforcement (`adjoin_root`):** Given a polynomial $g \in L[y]$
  irreducible over $L$ with degree $e \ge 2$, the extension algebra $M_0 = L[y]/(g(y))$ has dimension $n = [L:\mathbb{Q}]
  \cdot e$ over $\mathbb{Q}$ with basis $\{ \alpha^i \beta^j : 0 \le i < [L:\mathbb{Q}], 0 \le j < e \}$ (where $\alpha$
  is $L$'s generator and $\beta$ is the residue class of $y$). The algorithm searches for a primitive element $\gamma' =
  \beta + c\alpha$ over $c = 1, -1, 2, -2, \dots$ across a bounded search of up to 64 candidates. The $n \times n$
  rational multiplication matrix of $\gamma'$ on this basis is constructed, and its minimal polynomial $h'$ is computed
  via Frobenius normal form ([`nimblecas.frobenius`](frobenius.md)). The element $\gamma'$ is primitive if and only if
  $\deg(h') == n$. When $M_0$ is a field, $h'$ is automatically $\mathbb{Q}$-irreducible; if $g$ were not actually
  irreducible over $L$, $M_0$ would be a direct product of smaller fields, and any degree-$n$ minimal polynomial would be
  reducible over $\mathbb{Q}$. When `NumberField::create(h')` constructs the field, it verifies irreducibility via
  `factor_over_Q` and honestly rejects reducible polynomials with `MathError::domain_error`. The images of $\alpha$ and
  $\beta$ in $M = \mathbb{Q}[z]/(h')$ are recovered by solving the invertible linear system expressing powers $\gamma'^0,
  \dots, \gamma'^{n-1}$ against the target coordinates.
- **Mandatory post-construction vanishing checks:** Before returning an `AdjoinedRoot`, `adjoin_root` explicitly
  evaluates two algebraic relations in the newly constructed field $M$: $g(\operatorname{root}) == 0$ and
  $L.\operatorname{modulus}()(\operatorname{old\_generator}) == 0$. If either check fails to evaluate to zero, the
  function fails with `MathError::domain_error`.
- **Deterministic iterated splitting and root completeness guard (`splitting_field`):** Input irreducible factors are
  sorted into a canonical order (degree ascending, then coefficient-lexicographic) solely to make the field-growth
  sequence deterministic; roots are reported in the caller's original input factor order. The starting field is
  $\mathbb{Q}[x]/(f_0)$ where $f_0$ is the first non-linear factor in canonical order (or the trivial degree-1 field
  $\mathbb{Q}[t]/(t)$ if all inputs are already linear). In each outer pass, every input polynomial is re-embedded and
  re-factored over the current field via `factor_over_field`, and the first non-linear factor encountered is adjoined via
  `adjoin_root`. The loop terminates when every input factor splits into linear pieces. Before returning, the function
  verifies that each input factor contributes exactly $\deg(f_i)$ pairwise-distinct roots. A failure of this guard
  surfaces as `MathError::domain_error`.
- **Degree capping and practical envelope:** If $[L:\mathbb{Q}]$ would exceed `max_degree` at the initial field or at any
  subsequent adjunction, the function halts with `MathError::not_implemented` (for example, a quartic polynomial with
  Galois group $S_4$ requires degree 24; this module refuses rather than guess). Exhausting Trager's shift bound or the
  primitive-element search bound yields `MathError::not_implemented`. If `factor_over_Q` exhausts its internal search
  budget, `MathError::not_implemented` propagates. Any 64-bit integer overflow during rational arithmetic, resultant
  interpolation, or coordinate matrix operations produces `MathError::overflow`. The practical envelope of this module
  is roughly degree $\le 6$ with small coefficients. For problems requiring fields beyond this envelope (such as the
  degree-6 splitting field of $x^3 - 2$, which requires degree-9 and degree-18 Trager norms during iterated splitting),
  the unbounded bignum counterpart [`nimblecas.bigsplitfield`](bigsplitfield.md) should be used.

```cpp
import nimblecas.splitfield;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md), [`algnum`](algnum.md), [`algpoly`](algpoly.md), [`factor`](factor.md), [`resultant`](resultant.md), and [`frobenius`](frobenius.md).

## Data structures

```cpp
struct AdjoinedRoot {
    NumberField field;
    AlgebraicNumber old_generator;
    AlgebraicNumber root;
};

struct SplittingField {
    NumberField field;
    std::vector<std::pair<RationalPoly, std::vector<AlgebraicNumber>>> roots;
};
```

| Type | Member | Role |
| :--- | :--- | :--- |
| `AdjoinedRoot` | `field` | The simple extension field $M = \mathbb{Q}(\gamma') \cong L[y]/(g(y))$. |
| | `old_generator` | The image of $L$'s primitive generator $\alpha$ in $M$, required to re-embed elements of $L$ into $M$. |
| | `root` | The image of the adjoined root $\beta$ in $M$, satisfying $g(\operatorname{root}) = 0$. |
| `SplittingField` | `field` | The constructed splitting field containing all roots of the input polynomials. |
| | `roots` | Vector of pairs matching the caller's input order, mapping each input polynomial to its complete set of distinct roots in `field`. |

## Polynomial factorisation and roots over a number field

```cpp
[[nodiscard]] auto factor_over_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicPoly>>;

[[nodiscard]] auto roots_in_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicNumber>>;
```

| Function | Behaviour |
| :--- | :--- |
| `factor_over_field` | Factors a squarefree polynomial $f \in L[x]$ into its monic $L$-irreducible factors using Trager's norm algorithm. A nonzero constant polynomial returns an empty vector (no non-unit factors); a linear polynomial returns itself normalised to monic. Requires $f$ to live in `l` and $f \ne 0$. Returns `MathError::not_implemented` if no integer shift in $|s| \le 2[L:\mathbb{Q}]\deg(f)$ yields a squarefree norm or if `factor_over_Q` exceeds its budget; `MathError::overflow` if an `int64` rational boundary is crossed; `MathError::domain_error` if the product reconstruction fails. |
| `roots_in_field` | Extracts all roots of $f$ that lie in $L$, read off the degree-1 factors of `factor_over_field(l, f)` as $\lambda = -c_0 / c_1$. Order matches `factor_over_field`'s deterministic factor order. Errors propagate directly from `factor_over_field`. |

### Trager's norm algorithm mechanics

For a squarefree polynomial $f \in L[x]$ with $L = \mathbb{Q}[t]/(h(t))$, Trager's algorithm transforms factorisation over
$L$ into factorisation over $\mathbb{Q}$:

1. **Deterministic shifting:** The algorithm tests shifts $s \in \{0, 1, -1, 2, -2, \dots\}$ and forms $g(x) = f(x - s\gamma)$,
   where $\gamma$ is the generator of $L$.
2. **Norm computation by resultant interpolation:** The norm $N(x) = \operatorname{Norm}_{L/\mathbb{Q}}(g(x))$ has degree
   $[L:\mathbb{Q}] \cdot \deg(f)$. It is constructed by sampling $g(x)$ at $\deg(N) + 1$ rational points $x_0 \in \{0, 1, -1, 2, -2, \dots\}$,
   evaluating the resultant $\operatorname{Res}_t(h(t), g(x_0, t))$ over $\mathbb{Q}[t]$, and interpolating the values via
   Lagrange interpolation.
3. **Squarefree test:** If $\gcd(N, N') \ne 1$, $N(x)$ is not squarefree, indicating that the shift $s$ was unlucky. The loop
   advances to the next candidate $s$. The search bound is $|s| \le 2[L:\mathbb{Q}]\deg(f)$; exceeding this bound yields
   `MathError::not_implemented`.
4. **Factor extraction:** The squarefree polynomial $N(x)$ is factored over $\mathbb{Q}$ using `factor_over_Q(N)`. For each
   monic irreducible factor $N_i$, the corresponding $L$-factor is computed as $\gcd_L(g(x), \operatorname{embed}_L(N_i))$.
5. **Back-shifting and normalisation:** Each factor is shifted back via $x \mapsto x + s\gamma$ and normalised to monic.
6. **Reconstruction check:** The product of the monic factors scaled by the leading coefficient of $f$ must equal $f$ exactly.

## Extension fields and root adjunction

```cpp
[[nodiscard]] auto adjoin_root(const NumberField& l, const AlgebraicPoly& g)
    -> Result<AdjoinedRoot>;
```

| Function | Behaviour |
| :--- | :--- |
| `adjoin_root` | Adjoins a root of an irreducible polynomial $g \in L[y]$ ($\deg(g) = e \ge 2$) to $L$, constructing the simple extension $M = \mathbb{Q}(\gamma') \cong L[y]/(g(y))$. Searches candidate primitive elements $\gamma' = \beta + c\alpha$ over $c \in \{1, -1, 2, -2, \dots\}$ up to 64 candidates. Returns `MathError::domain_error` if $g$ does not live in `l`, $\deg(g) < 2$, $g$ is reducible over $L$ (detected when $h'$ is reducible over $\mathbb{Q}$), or post-construction vanishing verification fails; `MathError::not_implemented` if candidate search is exhausted; `MathError::overflow` if an `int64` rational boundary is crossed. |

### Primitive-element search and coordinate recovery

To represent $L[y]/(g(y))$ as a simple number field $M = \mathbb{Q}[z]/(h'(z))$:

1. **Product basis:** With $d = [L:\mathbb{Q}]$ and $e = \deg(g)$, the vector space $M_0$ has dimension $n = d \cdot e$
   over $\mathbb{Q}$. Its basis consists of elements $\alpha^i \beta^j$ for $0 \le i < d$ and $0 \le j < e$, where $\alpha$
   is $L$'s generator and $\beta$ is the class of $y$.
2. **Multiplication matrix and minimal polynomial:** For each candidate primitive element $\gamma' = \beta + c\alpha$
   ($c \in \{1, -1, 2, -2, \dots\}$, up to 64 candidates), the $n \times n$ rational matrix representing multiplication by
   $\gamma'$ on this basis is constructed. The minimal polynomial $h'$ of this matrix is computed using
   [`nimblecas.frobenius`](frobenius.md). The element $\gamma'$ is primitive if and only if $\deg(h') == n$.
3. **Irreducibility verification:** If $g$ is genuinely irreducible over $L$, $M_0$ is an algebraic field, and $h'$ is
   irreducible over $\mathbb{Q}$. If $g$ were reducible over $L$, $M_0$ would be a direct product of fields; any degree-$n$
   minimal polynomial would be reducible over $\mathbb{Q}$. When `NumberField::create(h')` is invoked, it verifies the
   irreducibility of $h'$ via `factor_over_Q` and honestly returns `MathError::domain_error` if $h'$ is reducible.
4. **Change of basis:** The change-of-basis matrix $V$ is assembled with columns corresponding to the coordinates of
   $\gamma'^0, \dots, \gamma'^{n-1}$. Because $\gamma'$ is primitive, $V$ is non-singular. Solving $V X = [\operatorname{vec}(\alpha) \mid \operatorname{vec}(\beta)]$
   recovers $\alpha$ and $\beta$ as explicit rational polynomials in $\gamma'$.
5. **Post-construction verification:** Before returning, the implementation verifies that $g(\operatorname{root}) == 0$ and
   $L.\operatorname{modulus}()(\operatorname{old\_generator}) == 0$ identically in $M$.

## Splitting field construction

```cpp
[[nodiscard]] auto splitting_field(std::span<const RationalPoly> irreducibles,
                                    std::int64_t max_degree) -> Result<SplittingField>;
```

| Function | Behaviour |
| :--- | :--- |
| `splitting_field` | Constructs the splitting field for a batch of $\mathbb{Q}$-irreducible polynomials, returning the field and each polynomial's full set of roots in that field. Inputs are sorted into canonical order to ensure deterministic field growth; roots are returned in the caller's original sequence. Returns `MathError::domain_error` if an input polynomial has degree $< 1$ or if the post-construction completeness/distinctness check fails; `MathError::not_implemented` if field degree exceeds `max_degree` at any point, outer loop iteration cap is reached, or primitive search / factorisation limits are hit; `MathError::overflow` on arithmetic overflow. |

### Iterated adjunction and verification guard

1. **Input validation and ordering:** Each polynomial in `irreducibles` must have degree $\ge 1$. The polynomials are sorted
   into canonical order (degree ascending, then coefficient-lexicographic, constant term first) to ensure a deterministic
   sequence of field extensions.
2. **Initial field:** If every polynomial is linear, the base field is initialised to the trivial degree-1 field
   $\mathbb{Q}[t]/(t)$. Otherwise, the initial field is $\mathbb{Q}[x]/(f_0)$, where $f_0$ is the first non-linear factor in
   canonical order. If its degree exceeds `max_degree`, execution halts immediately with `MathError::not_implemented`.
3. **Iterated factorisation loop:** An outer loop bounded by an iteration cap ($\max(\text{max\_degree}, 1) + m + 8$, where
   $m$ is the number of input polynomials) re-embeds and re-factors all input polynomials over the current field using
   `factor_over_field`. If any factor of degree $\ge 2$ remains, the first such factor is selected. If adjoining it would
   cause $[L:\mathbb{Q}] \cdot \deg(\text{piece})$ to exceed `max_degree`, the function returns `MathError::not_implemented`.
   Otherwise, `adjoin_root` extends the field, and the loop repeats.
4. **Completeness and distinctness guard:** Once all factors split into linear pieces, the roots are extracted as $-c_0$ from
   each monic linear piece $x + c_0$. The function verifies two invariants for each input polynomial:
   - Exactly $\deg(f_i)$ roots are obtained.
   - All extracted roots for that polynomial are pairwise distinct (`!roots[a].is_equal(roots[b])`).
   Failure of either check indicates an internal bug and produces `MathError::domain_error`.

## Error model

| Condition | Error |
| :--- | :--- |
| `factor_over_field`: `f` does not live in `l`, or `f` is the zero polynomial | `MathError::domain_error` |
| `factor_over_field`: Trager's shift bound $|s| \le 2[L:\mathbb{Q}]\deg(f)$ exhausted without finding a squarefree norm | `MathError::not_implemented` |
| `factor_over_field`: `factor_over_Q(N)` exceeds internal Kronecker budget | `MathError::not_implemented` |
| `factor_over_field`: norm degree mismatch or reconstructed product fails to equal $f$ | `MathError::domain_error` |
| `adjoin_root`: `g` does not live in `l`, or $\deg(g) < 2$ | `MathError::domain_error` |
| `adjoin_root`: primitive-element search exhausted (64 candidate shifts tested without degree-$n$ minimal polynomial) | `MathError::not_implemented` |
| `adjoin_root`: minimal polynomial $h'$ is reducible over $\mathbb{Q}$ (signalling $g$ was reducible over $L$) | `MathError::domain_error` |
| `adjoin_root`: post-construction vanishing check fails ($g(\operatorname{root}) \ne 0$ or $h(\operatorname{old\_generator}) \ne 0$) | `MathError::domain_error` |
| `splitting_field`: an input polynomial has non-positive degree ($\deg(p) < 1$) | `MathError::domain_error` |
| `splitting_field`: extension field degree exceeds `max_degree` (at initial field or any subsequent adjunction) | `MathError::not_implemented` |
| `splitting_field`: outer adjunction loop exceeds iteration cap | `MathError::not_implemented` |
| `splitting_field`: completeness guard fails (factor does not yield exactly $\deg(f_i)$ pairwise-distinct roots) | `MathError::domain_error` |
| Any arithmetic operation exceeds `int64` rational boundaries (degree multiplication, resultant, interpolation, matrix solve) | `MathError::overflow` |

## Worked examples

The following examples demonstrate polynomial factorisation over extension fields, root adjunction, and splitting-field
construction, adapted directly from the module's test suite (`tests/splitfield_tests.cpp`):

```cpp
import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.algnum;
import nimblecas.algpoly;
import nimblecas.splitfield;

using nimblecas::AlgebraicNumber;
using nimblecas::AlgebraicPoly;
using nimblecas::adjoin_root;
using nimblecas::factor_over_field;
using nimblecas::MathError;
using nimblecas::NumberField;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::roots_in_field;
using nimblecas::splitting_field;

// Helper to construct a RationalPoly from ascending integer coefficients.
auto rpoly = [](std::vector<std::int64_t> asc) -> RationalPoly {
    std::vector<Rational> c;
    c.reserve(asc.size());
    for (std::int64_t v : asc) {
        c.push_back(Rational::from_int(v));
    }
    return RationalPoly::from_coeffs(std::move(c));
};

// 1. Factorisation and root extraction over an extension field:
// x^2 - 2 splits over Q(sqrt2) = Q[t]/(t^2 - 2) into linear factors (x - sqrt2)(x + sqrt2).
const RationalPoly minpoly_q2 = rpoly({-2, 0, 1});  // t^2 - 2
auto q2 = NumberField::create(minpoly_q2).value();
const AlgebraicPoly f = AlgebraicPoly::embed(q2, minpoly_q2);

auto facs = factor_over_field(q2, f).value();
facs.size() == 2;
facs[0].degree() == 1 && facs[1].degree() == 1;

auto rts = roots_in_field(q2, f).value();
rts.size() == 2;
auto sq = rts[0].multiply(rts[0]).value();
sq.is_equal(q2.from_rational(Rational::from_int(2)));  // root^2 == 2

// 2. Simple extension via primitive-element search:
// Adjoining a root of x^2 - 3 to Q(sqrt2) constructs Q(sqrt2, sqrt3) of degree 4.
const AlgebraicPoly g = AlgebraicPoly::embed(q2, rpoly({-3, 0, 1}));  // x^2 - 3
auto adj = adjoin_root(q2, g).value();
adj.field.degree() == 4;

// The image of the old generator squares to 2, and the adjoined root squares to 3:
auto og2 = adj.old_generator.multiply(adj.old_generator).value();
og2.is_equal(adj.field.from_rational(Rational::from_int(2)));  // old_generator^2 == 2

auto rt2 = adj.root.multiply(adj.root).value();
rt2.is_equal(adj.field.from_rational(Rational::from_int(3)));  // root^2 == 3

// 3. Splitting field of the cyclic cubic x^3 - 3x - 1:
// The Galois group is A3 (cyclic of order 3), so all three roots lie in the degree-3 field.
std::vector<RationalPoly> cyclic_cubic{rpoly({-1, -3, 0, 1})};
auto sf_cyclic = splitting_field(cyclic_cubic, /*max_degree=*/12).value();
sf_cyclic.field.degree() == 3;
const auto& cyclic_roots = sf_cyclic.roots[0].second;
cyclic_roots.size() == 3;

// Roots are pairwise distinct:
!cyclic_roots[0].is_equal(cyclic_roots[1]) &&
!cyclic_roots[0].is_equal(cyclic_roots[2]) &&
!cyclic_roots[1].is_equal(cyclic_roots[2]);

// 4. Honest refusal when degree bound is exceeded:
// x^3 - 2 has splitting field of degree 6; max_degree = 2 refuses honestly.
std::vector<RationalPoly> x3_minus_2{rpoly({-2, 0, 0, 1})};
auto sf_capped = splitting_field(x3_minus_2, /*max_degree=*/2);
sf_capped.error() == MathError::not_implemented;

// 5. The int64 arithmetic ceiling:
// Splitting x^3 - 2 completely requires degree-9 and degree-18 Trager norms. Under int64
// Rational arithmetic, this boundary may encounter overflow:
auto sf_cubic = splitting_field(x3_minus_2, /*max_degree=*/12);
if (!sf_cubic) {
    // Returns MathError::not_implemented or MathError::overflow honestly when int64 saturates.
    bool honest_failure = (sf_cubic.error() == MathError::not_implemented ||
                           sf_cubic.error() == MathError::overflow);
}
```

## See also

- [`nimblecas.bigsplitfield`](bigsplitfield.md) — the unbounded, `BigRational`-backed counterpart that builds
  splitting fields without `int64` overflow ceilings (such as the full degree-6 splitting field of $x^3 - 2$).
- [`nimblecas.algnum`](algnum.md) — exact arithmetic in simple algebraic number fields `NumberField` / `AlgebraicNumber`.
- [`nimblecas.algpoly`](algpoly.md) — dense univariate polynomials over a single algebraic number field `AlgebraicPoly`.
- [`nimblecas.ratpoly`](ratpoly.md) — rational arithmetic and univariate polynomials over $\mathbb{Q}$.
- [`nimblecas.factor`](factor.md) — exact polynomial factorisation over $\mathbb{Q}$ (`factor_over_Q`).
- [`nimblecas.resultant`](resultant.md) — polynomial resultant used to evaluate field norms without normalisation.
- [`nimblecas.frobenius`](frobenius.md) — minimal polynomial calculation via Frobenius form in `adjoin_root`.
- [`nimblecas.jordan`](jordan.md) — exact Jordan canonical form computation over splitting fields.
- [Documentation hub](../Index.md)
