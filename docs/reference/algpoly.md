# `nimblecas.algpoly` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/algpoly/algpoly.cppm`

Dense **univariate polynomials over a single algebraic extension field** `L =
Q(alpha) = Q[t]/(m(t))`, for a monic irreducible `m` (see
[`algnum`](algnum.md)). `AlgebraicPoly` is the shared arithmetic substrate for
two computations that need polynomial arithmetic over an extension field rather
than over `Q` itself:

- **Rothstein–Trager over `Q(alpha)`** ([`rothstein`](rothstein.md)): the
  logarithm argument `v(x) = gcd_x(D, A − alpha·D')` for an irrational/complex
  residue is a gcd of two `L[x]` polynomials;
- **Jordan form over a splitting field** ([`jordan`](jordan.md)): the
  eigenvalue bookkeeping over `Q(alpha)` manipulates `L[x]`.

```cpp
import nimblecas.algpoly;
```

Depends on [`core`](core.md) (`Result` / `MathError`), [`ratpoly`](ratpoly.md)
(`RationalPoly`, the `Q[x]` polynomial an `AlgebraicPoly` can be lifted from),
and [`algnum`](algnum.md) (`NumberField` / `AlgebraicNumber`, the field every
coefficient lives in).

## Honesty boundary — exact over the field, never a mismatched field

`AlgebraicPoly` is **exact over `L`**, no numeric fallback anywhere: every
coefficient is an `AlgebraicNumber` residue in `Q[t]/(m)`, and every ring
operation is built from `AlgebraicNumber`'s exact, overflow-checked
`add`/`subtract`/`multiply`/`inverse`. Because `L` is a **field** — every
nonzero element is invertible via extended Euclid in `Q[t]` — general
Euclidean polynomial division needs only the divisor's leading-coefficient
inverse, so `divide`, `monic`, and the Euclidean `gcd` are always exact and
always terminate; there is no notion of an "inexact" `AlgebraicPoly` result.

The one thing this module refuses to paper over is **mixing fields**:
combining two `AlgebraicPoly` values built over different `NumberField`
instances (different minimal polynomials, or the same minimal polynomial
constructed twice) is `MathError::domain_error`, never a silent coercion or a
polynomial whose coefficients quietly disagree on what `alpha` means.
Underlying `int64` rational arithmetic (inherited from `AlgebraicNumber` /
`RationalPoly`) is overflow-checked throughout and surfaces
`MathError::overflow` rather than wrapping; dividing (or taking `monic()` or a
leading-coefficient inverse of) the zero polynomial is
`MathError::division_by_zero`.

## Representation

A dense coefficient vector, `coeffs[i]` the coefficient of `x^i`, every
coefficient an `AlgebraicNumber` in the **same** field (`is_same` compares
minimal polynomials). Trailing zero coefficients are trimmed, so the zero
polynomial has an empty vector and `degree()` is `-1` — the same convention
`RationalPoly` uses. Because `AlgebraicNumber` has no default constructor (its
residue always names a field), coefficient vectors are always built by
`push_back` / `field.zero()` fill, never `resize()`.

## API

All entry points are member functions of `AlgebraicPoly` in namespace
`nimblecas`, `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `zero` | `static auto zero(const NumberField& field) -> AlgebraicPoly` | The zero polynomial over `field` (empty coefficients, degree `-1`). |
| `from_coeffs` | `static auto from_coeffs(const NumberField& field, std::vector<AlgebraicNumber> coeffs) -> Result<AlgebraicPoly>` | Build from coefficients (index `i` = coefficient of `x^i`). Every coefficient must live in `field` — a mismatch is `domain_error`. Trailing zeros are trimmed. |
| `embed` | `static auto embed(const NumberField& field, const RationalPoly& p) -> AlgebraicPoly` | Lift a `Q[x]` polynomial into `L[x]` via `NumberField::from_rational` on each coefficient (infallible). Trailing zeros are trimmed. |
| `field` | `auto field() const -> const NumberField&` | The field `L` every coefficient lives in. |
| `is_zero` | `auto is_zero() const -> bool` | `true` iff the coefficient vector is empty. |
| `degree` | `auto degree() const -> std::int64_t` | `-1` for the zero polynomial, else the highest nonzero-coefficient index. |
| `coefficients` | `auto coefficients() const -> std::span<const AlgebraicNumber>` | The trimmed coefficient vector. |
| `coefficient` | `auto coefficient(std::size_t i) const -> AlgebraicNumber` | Coefficient of `x^i`; `field().zero()` beyond the stored degree. |
| `leading_coefficient` | `auto leading_coefficient() const -> AlgebraicNumber` | Highest-degree coefficient; `field().zero()` for the zero polynomial. |
| `is_equal` | `auto is_equal(const AlgebraicPoly& o) const -> bool` | Same field **and** identical trimmed coefficient vectors. |
| `add` / `subtract` | `auto add(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>` | Coefficientwise; field mismatch → `domain_error`. |
| `multiply` | `auto multiply(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>` | Convolution over `L`; field mismatch → `domain_error`. |
| `scale` | `auto scale(const AlgebraicNumber& c) const -> Result<AlgebraicPoly>` | Every coefficient times the field element `c` (which must live in the same field). |
| `derivative` | `auto derivative() const -> Result<AlgebraicPoly>` | Formal `d/dx`; a constant or the zero polynomial differentiates to `0`. |
| `evaluate` | `auto evaluate(const AlgebraicNumber& x) const -> Result<AlgebraicNumber>` | Horner evaluation at a field element `x` (must live in the same field). |
| `divide` | `auto divide(const AlgebraicPoly& o) const -> Result<AlgPolyDivMod>` | Euclidean division: `*this == quotient*o + remainder`, `deg(remainder) < deg(o)`. Dividing by zero is `division_by_zero`. |
| `monic` | `auto monic() const -> Result<AlgebraicPoly>` | Normalise to leading coefficient `1`. The zero polynomial has no monic form: `division_by_zero`. |
| `gcd` | `auto gcd(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>` | Monic Euclidean gcd (remainders re-normalised to monic each step to bound coefficient growth); `gcd(0, 0) == 0`. |
| `to_string` | `auto to_string(std::string_view var = "x", std::string_view alpha_var = "a") const -> std::string` | Human-readable rendering, e.g. `"(a)*x + (1)"`.  |

### `AlgPolyDivMod`

```cpp
struct AlgPolyDivMod {
    AlgebraicPoly quotient;
    AlgebraicPoly remainder;
};
```

## Error model

| Condition | Error |
| :--- | :--- |
| Combining (`add`/`subtract`/`multiply`/`scale`/`evaluate`/`divide`/`gcd`) two values whose `NumberField`s differ | `MathError::domain_error` |
| `from_coeffs` given a coefficient not in `field` | `MathError::domain_error` |
| `divide` / `monic` / a leading-coefficient inverse against the zero polynomial | `MathError::division_by_zero` |
| Any underlying `int64` rational numerator/denominator overflow | `MathError::overflow` |

## Worked example — `(x − alpha)(x + alpha) = x^2 + 1` over `Q(i)`

```cpp
import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.algnum;
import nimblecas.algpoly;
using namespace nimblecas;

// Q(i) = Q[t]/(t^2 + 1), alpha^2 = -1.
auto x2_plus_1 = RationalPoly::from_coeffs(
    {Rational::from_int(1), Rational::from_int(0), Rational::from_int(1)});
const NumberField qi = NumberField::create(x2_plus_1).value();
const AlgebraicNumber alpha = qi.generator().value();

// (x - alpha) and (x + alpha), as AlgebraicPoly over Q(i).
const AlgebraicPoly minus =
    AlgebraicPoly::from_coeffs(qi, {alpha.negate().value(), qi.one()}).value();  // x - alpha
const AlgebraicPoly plus =
    AlgebraicPoly::from_coeffs(qi, {alpha, qi.one()}).value();                    // x + alpha

// The load-bearing identity: (x - alpha)(x + alpha) = x^2 - alpha^2 = x^2 + 1.
const AlgebraicPoly prod = minus.multiply(plus).value();
const AlgebraicPoly target = AlgebraicPoly::embed(qi, x2_plus_1);
prod.is_equal(target);   // true

// Division and gcd agree: (x^2+1) / (x - alpha) == (x + alpha) rem 0.
auto dm = target.divide(minus).value();
dm.remainder.is_zero();              // true
dm.quotient.is_equal(plus);          // true
target.gcd(minus).value().is_equal(minus);  // true (x - alpha is already monic)

// alpha is a root: evaluate(alpha) == 0 for both the linear factor and the target.
minus.evaluate(alpha).value().is_zero();   // true
target.evaluate(alpha).value().is_zero();  // true

// derivative of x^2 + 1 is 2x.
auto d = target.derivative().value();
d.degree() == 1;                                       // true
d.coefficient(1).is_equal(qi.from_rational(Rational::from_int(2)));  // true

// Mixing fields is an honest error, never a silent coercion.
auto q2 = NumberField::create(RationalPoly::from_coeffs(
    {Rational::from_int(-2), Rational::from_int(0), Rational::from_int(1)})).value();  // Q(sqrt2)
auto beta = q2.generator().value();
auto other = AlgebraicPoly::from_coeffs(q2, {beta, q2.one()}).value();  // x + sqrt2, over Q(sqrt2)
minus.add(other).error();               // MathError::domain_error
minus.divide(AlgebraicPoly::zero(qi)).error();  // MathError::division_by_zero
```

## See also

- [`nimblecas.algnum`](algnum.md) — the `Q(alpha) = Q[t]/(m)` extension field
  and its `AlgebraicNumber` arithmetic every coefficient here lives in.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`) an `AlgebraicPoly` can be lifted from via `embed`.
- [`nimblecas.rothstein`](rothstein.md) — `log_part_extended` builds the
  `A − alpha·D'` gcd over `L[x]` on this module for irrational/complex
  residues.
- [`nimblecas.jordan`](jordan.md) — Jordan canonical form over a quadratic
  extension `Q(alpha)` (`jordan_form`), the other consumer of this substrate.
- [Documentation hub](../Index.md)
