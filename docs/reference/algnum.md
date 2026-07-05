# `nimblecas.algnum` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/algnum/algnum.cppm`

Exact arithmetic in a **simple algebraic number field**
`Q(alpha) = Q[x]/(m(x))`, where `m` is a **monic irreducible** polynomial over
`Q` of degree `d >= 1`. This is the algebraic-number substrate a
Jordan-canonical-form module builds on: it represents an irrational or complex
**algebraic** number exactly as a residue class of [`RationalPoly`](ratpoly.md) —
`sqrt(2)` is the class of `x` in `Q[x]/(x^2 - 2)`, `i` is the class of `x` in
`Q[x]/(x^2 + 1)`, `cbrt(2)` is the class of `x` in `Q[x]/(x^3 - 2)`. Every element
is the exact rational-coefficient polynomial it mathematically is — never a
floating-point stand-in — so downstream eigenvalue / canonical-form work over these
fields stays exact.

```cpp
import nimblecas.algnum;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md),
[`factor`](factor.md) (irreducibility check at field construction), and
[`matrix`](matrix.md) (norm / trace via the multiplication map).

## The irreducibility precondition (enforced)

A `NumberField` exists **only** for a monic irreducible `m`; this is not a
documented assumption but a checked invariant. `NumberField::create` normalises
`m` to monic (dividing by its leading coefficient — same ideal, same field) and
then **verifies irreducibility** by calling
[`factor_over_Q`](factor.md): `m` is accepted iff its factorization is a single
irreducible factor of multiplicity 1. A reducible `m` (e.g. `x^2 - 1 =
(x-1)(x+1)`), a perfect power (e.g. `(x-1)^2`), a nonzero constant, or the zero
polynomial all yield `MathError::domain_error`. Because reducibility is rejected
up front, `Q[x]/(m)` is always a genuine **field**: every nonzero element is
invertible, so `inverse`/`divide` never silently return a wrong value for a
non-invertible element (no such element exists).

## The honesty / overflow contract

Following the rest of the engine (Rule 32), every fallible operation returns
`Result<T> = std::expected<T, MathError>`; nothing throws and nothing returns a
plausible-but-wrong value. All arithmetic is **exact** and **overflow-checked**,
inherited from `Rational` / `RationalPoly`: an `int64` numerator or denominator
that would overflow surfaces as `MathError::overflow` rather than wrapping.

| Situation | Result |
| :--- | :--- |
| `m` reducible / constant / zero | `domain_error` (at `create`) |
| `inverse`/`divide` of/by the zero element | `division_by_zero` |
| binary op on elements of **different** fields | `domain_error` |
| `pow` with a negative exponent | `domain_error` |
| `int64` overflow in the exact arithmetic | `overflow` |
| irreducibility search exceeds `factor_over_Q`'s budget | `not_implemented` (propagated) |

## `NumberField` — `Q(alpha) = Q[x]/(m(x))`

The field carries its monic irreducible minimal polynomial `m` (behind a
[`CowPtr`](core.md) so copies are O(1)) and acts as the factory for its elements.

### Construction & observers

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `create` | `static auto create(const RationalPoly& minimal) -> Result<NumberField>` | Normalise `minimal` to monic and **verify irreducibility**; `domain_error` if reducible / constant / zero. |
| `degree` | `auto degree() const -> std::int64_t` | `d = deg m` (the extension degree `[Q(alpha):Q]`). |
| `modulus` | `auto modulus() const -> const RationalPoly&` | The monic irreducible minimal polynomial `m`. |
| `is_same` | `auto is_same(const NumberField& o) const -> bool` | Same field iff the minimal polynomials are identical. |
| `to_string` | `auto to_string(std::string_view var = "a") const -> std::string` | e.g. `"Q[a]/(...)"`. |

### Element factories

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `zero` | `auto zero() const -> AlgebraicNumber` | The additive identity `0`. |
| `one` | `auto one() const -> AlgebraicNumber` | The multiplicative identity `1`. |
| `from_rational` | `auto from_rational(const Rational& c) const -> AlgebraicNumber` | The constant `c`. |
| `generator` | `auto generator() const -> Result<AlgebraicNumber>` | `alpha = x mod m`. Fails only on overflow. |
| `from_poly` | `auto from_poly(const RationalPoly& p) const -> Result<AlgebraicNumber>` | The class of `p`, i.e. `p reduced mod m`. Fails only on overflow. |

## `AlgebraicNumber` — an element of a `NumberField`

Holds its field and its **canonical residue**: a `RationalPoly` of degree `< d`,
the unique remainder mod `m`. Because `{1, alpha, ..., alpha^(d-1)}` is a `Q`-basis
of the field, this residue is a normal form and equality is a coefficient-vector
compare.

### Observers

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `field` | `auto field() const -> const NumberField&` | The owning field. |
| `value` | `auto value() const -> const RationalPoly&` | The canonical residue (coordinates of `1, alpha, ..., alpha^(d-1)`). |
| `is_zero` | `auto is_zero() const -> bool` | `true` for the zero element. |
| `is_one` | `auto is_one() const -> bool` | `true` for the multiplicative identity. |
| `is_equal` | `auto is_equal(const AlgebraicNumber& o) const -> bool` | Same field **and** same residue. |
| `to_string` | `auto to_string(std::string_view var = "a") const -> std::string` | Renders the residue polynomial. |

### Field operations (all exact; overflow-checked)

```cpp
[[nodiscard]] auto add(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
[[nodiscard]] auto subtract(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
[[nodiscard]] auto negate() const -> Result<AlgebraicNumber>;
[[nodiscard]] auto multiply(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
[[nodiscard]] auto inverse() const -> Result<AlgebraicNumber>;
[[nodiscard]] auto divide(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
[[nodiscard]] auto pow(std::int64_t exponent) const -> Result<AlgebraicNumber>;
[[nodiscard]] auto norm() const -> Result<Rational>;
[[nodiscard]] auto trace() const -> Result<Rational>;
```

- **`add` / `subtract` / `negate`** act componentwise on the residue (a sum of two
  degree-`<d` polynomials is already reduced).
- **`multiply`** is a polynomial product reduced mod `m`.
- **`inverse`** uses the **extended Euclidean algorithm** in `Q[x]`: since `m` is
  irreducible and `0 != a` has `deg a < d`, `gcd(a, m)` is a nonzero constant `g`
  with `u*a + v*m = g`, whence `a^{-1} = (u / g) mod m`. The zero element is
  `division_by_zero`.
- **`divide`** is `a * b^{-1}` (`division_by_zero` when `b == 0`).
- **`pow`** is non-negative-exponent repeated squaring (`pow(0) == 1`); a negative
  exponent is `domain_error`.
- **`norm`** and **`trace`** are the exact determinant and trace of the `Q`-linear
  *multiplication-by-`a`* map on the basis `{1, alpha, ..., alpha^(d-1)}` (built as a
  `d x d` rational [`Matrix`](matrix.md)). For `a` in `Q(alpha)` of degree `d`, the norm
  is `(-1)^d` times the constant term, and the trace is the negated subleading
  coefficient, of that map's characteristic polynomial. (The `(-1)^d` sign matters for
  odd `d`: e.g. `norm(cbrt2) = +2` while `x^3 - 2` has constant term `-2`.)

> `minimal_polynomial(element)` is **not** provided in this module (it is optional
> for the substrate); the multiplication-map matrix returned internally by norm/trace
> is exactly the object whose characteristic polynomial would yield it, so it can be
> added later without changing the representation.

## Example

```cpp
import nimblecas.algnum;
import nimblecas.ratpoly;
using namespace nimblecas;

// Q(sqrt2) = Q[x]/(x^2 - 2).
const auto F = NumberField::create(
    RationalPoly::from_coeffs({Rational::from_int(-2), Rational::from_int(0),
                               Rational::from_int(1)})).value();
const auto a   = F.generator().value();      // sqrt2
const auto one = F.one();

auto two   = a.multiply(a).value();          // sqrt2 * sqrt2 = 2
auto minus = one.add(a).value()              // (1 + sqrt2)(1 - sqrt2) = -1
                .multiply(one.subtract(a).value()).value();
auto inv   = one.add(a).value().inverse().value();  // (1 + sqrt2)^{-1} = sqrt2 - 1
auto N     = one.add(a).value().norm().value();     // N(1 + sqrt2) = -1
auto T     = one.add(a).value().trace().value();    // Tr(1 + sqrt2) = 2

// i^{-1} = -i in Q(i) = Q[x]/(x^2 + 1).
const auto G = NumberField::create(
    RationalPoly::from_coeffs({Rational::from_int(1), Rational::from_int(0),
                               Rational::from_int(1)})).value();
auto iinv = G.generator().value().inverse().value();  // -i
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the `Rational` / `RationalPoly` element
  representation this module reduces mod `m`.
- [`nimblecas.factor`](factor.md) — factorization over `Q`, used to verify the
  irreducibility precondition.
- [`nimblecas.matrix`](matrix.md) — the exact rational matrix behind norm / trace.
- [Documentation hub](../Index.md)
```
