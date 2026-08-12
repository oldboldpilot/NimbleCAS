# `nimblecas.bigalgpoly` — Polynomials over an algebraic number field, unbounded (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigalgpoly/bigalgpoly.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.algpoly`](algpoly.md).
A `BigAlgebraicPoly` is a polynomial in `x` whose coefficients are elements of
**one** [`BigNumberField`](bigalgnum.md) `L = Q(alpha) = Q[t]/(m(t))`. It is
the arbitrary-precision analogue of `algpoly`'s `AlgebraicPoly`: where
`AlgebraicPoly` carries `int64` `AlgebraicNumber` coefficients and surfaces
`MathError::overflow` the moment a residue coefficient saturates,
`BigAlgebraicPoly` carries `BigAlgebraicNumber` coefficients
(`BigRationalPoly` residues) and has **no such ceiling**. It is the natural
home for the two computations that need polynomial arithmetic over an
extension field beyond the `int64` envelope: Rothstein-Trager's log argument
`v(x) = gcd_x(D, A - alpha*D')` (a gcd of two `L[x]` polynomials), and
Trager's norm factorization / eigenvalue bookkeeping for Jordan over a
splitting field.

**Representation.** A dense coefficient vector, `coeffs_[i]` the coefficient
of `x^i`, every coefficient a `BigAlgebraicNumber` in the **same** field
(equal minimal polynomials). Trailing zero coefficients are trimmed, so the
zero polynomial has an empty vector and `degree()` is `-1` (matching
`BigRationalPoly`'s convention).

**Field.** `L` is a field: every nonzero `BigAlgebraicNumber` is invertible
(extended Euclid in `Q[t]`). Hence general Euclidean division works with only
the divisor's leading-coefficient inverse, and monic gcd is the plain
Euclidean algorithm; remainders are re-normalised to monic every step to
bound coefficient growth. Should the modulus be reducible, a residue sharing
a factor with it surfaces `MathError::division_by_zero` from
`inverse()` — inherited honestly from [`BigAlgebraicNumber`](bigalgnum.md)
rather than fabricating a wrong value.

Every fallible operation returns `Result<T>`; nothing throws and nothing
returns a plausible-but-wrong value. Combining polynomials over different
fields is `MathError::domain_error`; dividing (or taking monic / the leading
inverse) by the zero polynomial is `MathError::division_by_zero`. Because the
underlying `BigRational` / `BigRationalPoly` arithmetic only combines
magnitudes, it **cannot overflow** — `MathError::overflow` can **never**
arise in this tier, which is the entire point of the bignum mirror.

```cpp
import nimblecas.bigalgpoly;
```

Depends on [`core`](core.md), `bigrational`, [`bigratpoly`](bigratpoly.md),
and [`bigalgnum`](bigalgnum.md).

## The honesty boundary

- **Exact over an unbounded rational tier — `MathError::overflow` can never
  be produced.** Every ring operation (`add`, `subtract`, `multiply`,
  `scale`, `derivative`, `evaluate`) reduces to `BigAlgebraicNumber`
  arithmetic, which cannot overflow.
- **Field mismatch is `domain_error`.** Every binary operation checks that
  both operands (and any scalar or evaluation point) live in the same field.
- **Division by the zero polynomial, or taking `monic()` of it, is
  `division_by_zero`** — never a fabricated quotient.
- **Inherited zero-divisor honesty.** If `L`'s modulus is reducible and a
  coefficient's inverse is required (leading-coefficient inverse in `divide`
  / `monic`), a zero-divisor coefficient surfaces `division_by_zero` from
  `BigAlgebraicNumber::inverse()`, propagated rather than masked.

## API summary

```cpp
[[nodiscard]] static auto zero(const BigNumberField& field) -> BigAlgebraicPoly;
[[nodiscard]] static auto from_coeffs(const BigNumberField& field,
                                      std::vector<BigAlgebraicNumber> coeffs)
    -> Result<BigAlgebraicPoly>;
[[nodiscard]] static auto embed(const BigNumberField& field, const BigRationalPoly& p)
    -> BigAlgebraicPoly;

[[nodiscard]] auto field() const -> const BigNumberField&;
[[nodiscard]] auto is_zero() const -> bool;
[[nodiscard]] auto degree() const -> std::int64_t;
[[nodiscard]] auto coefficients() const -> std::span<const BigAlgebraicNumber>;
[[nodiscard]] auto coefficient(std::size_t i) const -> BigAlgebraicNumber;
[[nodiscard]] auto leading_coefficient() const -> BigAlgebraicNumber;
[[nodiscard]] auto is_equal(const BigAlgebraicPoly& o) const -> bool;

// Ring operations (all exact; NEVER overflow; field-mismatch -> domain_error)
[[nodiscard]] auto add(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
[[nodiscard]] auto subtract(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
[[nodiscard]] auto multiply(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
[[nodiscard]] auto scale(const BigAlgebraicNumber& c) const -> Result<BigAlgebraicPoly>;
[[nodiscard]] auto derivative() const -> Result<BigAlgebraicPoly>;
[[nodiscard]] auto evaluate(const BigAlgebraicNumber& x) const -> Result<BigAlgebraicNumber>;

// Euclidean division: divide by zero -> division_by_zero.
[[nodiscard]] auto divide(const BigAlgebraicPoly& o) const -> Result<BigAlgPolyDivMod>;
// Normalise to monic; the zero polynomial -> division_by_zero.
[[nodiscard]] auto monic() const -> Result<BigAlgebraicPoly>;
// Monic Euclidean gcd; gcd(0, 0) == 0.
[[nodiscard]] auto gcd(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;

[[nodiscard]] auto to_string(std::string_view var = "x",
                             std::string_view alpha_var = "a") const -> std::string;
```

```cpp
struct BigAlgPolyDivMod {
    BigAlgebraicPoly quotient;
    BigAlgebraicPoly remainder;
};
```

## Error model

| Condition | Error |
| :--- | :--- |
| `from_coeffs` with a coefficient from a different field | `MathError::domain_error` |
| `add` / `subtract` / `multiply` / `scale` / `evaluate` across different fields | `MathError::domain_error` |
| `divide` by the zero polynomial | `MathError::division_by_zero` |
| `monic()` of the zero polynomial | `MathError::division_by_zero` |
| A coefficient inverse required by `divide` / `monic` that is a zero divisor of a reducible modulus | `MathError::division_by_zero` |

There is **no `overflow` row**: `BigRational` / `BigRationalPoly` /
`BigAlgebraicNumber` arithmetic cannot wrap, so the `overflow` the `int64`
[`algpoly`](algpoly.md) can raise on saturated coefficients is unreachable
here.

## Worked example

```cpp
import nimblecas.bigalgpoly;
import nimblecas.bigalgnum;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
using namespace nimblecas;

auto q  = [](std::int64_t v) { return BigRational::from_int(v); };
auto qs = [](std::string_view s) { return BigRational::from_string(s).value(); };

// Q(sqrt2) = Q[t]/(t^2 - 2). B = 10^10.
const BigNumberField q2 =
    BigNumberField::create(BigRationalPoly::from_coeffs({q(-2), q(0), q(1)})).value();

// (x + B*sqrt2)(x - B*sqrt2) == x^2 - 2*B^2 == x^2 - 2e20. The constant term -2e20 far
// exceeds int64's ~9.22e18 ceiling, exact only in the bignum tier.
auto p = BigAlgebraicPoly::from_coeffs(
    q2, {q2.from_poly(BigRationalPoly::monomial(qs("10000000000"), 1)).value(),
         q2.from_bigrational(q(1))}).value();               // B*sqrt2 + x
auto r = BigAlgebraicPoly::from_coeffs(
    q2, {q2.from_poly(BigRationalPoly::monomial(qs("10000000000").negate(), 1)).value(),
         q2.from_bigrational(q(1))}).value();                // -B*sqrt2 + x

auto prod = p.multiply(r).value();
prod.coefficient(0).is_equal(q2.from_bigrational(qs("-200000000000000000000")));  // -2e20

// It divides back out exactly, remainder 0.
auto dm = prod.divide(p).value();
dm.remainder.is_zero();          // true
dm.quotient.is_equal(r);         // x - B*sqrt2
```

## See also

- [`nimblecas.algpoly`](algpoly.md) — the `int64`-`AlgebraicNumber`,
  overflow-checked sibling this module big-backs; use it when residue
  coefficients fit `int64`.
- [`nimblecas.bigalgnum`](bigalgnum.md) — the `BigNumberField` /
  `BigAlgebraicNumber` coefficient type.
- [`nimblecas.bigratpoly`](bigratpoly.md) — the unbounded `Q[x]` polynomial
  tier that `embed()` lifts from.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field underlying every layer.
- [Documentation hub](../Index.md)
