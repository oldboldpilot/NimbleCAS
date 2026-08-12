# `nimblecas.bigalgnum` — Algebraic number fields over the unbounded rationals (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigalgnum/bigalgnum.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.algnum`](algnum.md).
Where `NumberField` / `AlgebraicNumber` carry `int64` `RationalPoly` residues
and surface `MathError::overflow` the moment a residue coefficient saturates,
`BigNumberField` / `BigAlgebraicNumber` carry
[`BigRationalPoly`](bigratpoly.md) residues and have **no such ceiling**: it
is the layer-2 field on which a bignum-backed splitting field can exceed the
`int64` overflow envelope. The public method surface mirrors `algnum`'s
exactly, so this is a drop-in bignum replacement.

A number field here is a simple extension `Q(alpha) = Q[x]/(m(x))` for a monic
minimal polynomial `m` of degree `d >= 1` (a non-monic `m` is normalised to
monic). An element is the unique `BigRationalPoly` residue of degree `< d`;
`{1, alpha, ..., alpha^(d-1)}` is a Q-basis, so the residue is a normal form
and equality is a coefficient-vector compare. Add / subtract / negate are
componentwise; multiply is a polynomial product reduced mod `m`; inverse of a
nonzero element uses the extended Euclidean algorithm over `BigRationalPoly`.

**Divergence from `algnum` (honest, documented).** `algnum` verifies
irreducibility of `m` via `factor_over_Q`, so `gcd(a, m)` is always a nonzero
constant and every nonzero element is invertible. There is **no bignum
polynomial-factoring layer available at this tier**, so `create()` does
**not** prove irreducibility — it only rejects the zero polynomial and
constants and normalises to monic. The honesty is preserved at **inversion
time** instead: if the modulus is reducible and the residue `a` shares a
nontrivial factor with it, `gcd(a, m)` has degree `>= 1` (not a unit), and
`inverse()` reports `MathError::division_by_zero` rather than fabricating a
wrong "inverse". Inverting the zero element is `division_by_zero` as well.

Every fallible operation returns `Result<T>`; nothing throws. Because
`BigRational` / `BigRationalPoly` arithmetic only combines magnitudes, it
**cannot overflow** — `MathError::overflow` never arises from the field
arithmetic, which is the entire point of this tier. Combining elements of
different fields is `MathError::domain_error`; a negative power is
`domain_error`. The modulus is held behind a `CowPtr`, so copying a field (and
every element that embeds it) is an O(1) refcount bump.

```cpp
import nimblecas.bigalgnum;
```

Depends on [`core`](core.md), `bigint`, `bigrational`, and
[`bigratpoly`](bigratpoly.md).

## The honesty boundary

- **Exact over an unbounded rational tier — `MathError::overflow` can never
  be produced.** Add, subtract, negate, scale, multiply, `pow`, `norm`, and
  `trace` only combine `BigRational` magnitudes.
- **`create()` does not prove irreducibility.** No bignum factoring layer
  exists at this tier, so a reducible modulus is silently accepted.
- **Honesty is enforced at `inverse()` instead.** A residue that is a zero
  divisor against a reducible modulus yields a non-unit `ext_gcd` (degree
  `>= 1`), which surfaces as `MathError::division_by_zero` — never a
  fabricated inverse.

## API summary

```cpp
// BigNumberField — Q[x]/(m). m normalised to monic; zero/constant m is domain_error.
[[nodiscard]] static auto create(const BigRationalPoly& minimal) -> Result<BigNumberField>;
[[nodiscard]] auto degree() const -> std::int64_t;
[[nodiscard]] auto modulus() const -> const BigRationalPoly&;
[[nodiscard]] auto is_same(const BigNumberField& o) const -> bool;

[[nodiscard]] auto zero() const -> BigAlgebraicNumber;
[[nodiscard]] auto one() const -> BigAlgebraicNumber;
[[nodiscard]] auto from_bigrational(const BigRational& c) const -> BigAlgebraicNumber;
[[nodiscard]] auto from_poly(const BigRationalPoly& p) const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto generator() const -> Result<BigAlgebraicNumber>;  // alpha = x mod m

// BigAlgebraicNumber — an element of a BigNumberField (residue, degree < d).
[[nodiscard]] auto field() const -> const BigNumberField&;
[[nodiscard]] auto value() const -> const BigRationalPoly&;
[[nodiscard]] auto is_zero() const -> bool;
[[nodiscard]] auto is_one() const -> bool;
[[nodiscard]] auto is_equal(const BigAlgebraicNumber& o) const -> bool;

[[nodiscard]] auto add(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto subtract(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto negate() const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto scale(const BigRational& s) const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto multiply(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
// Extended Euclid over BigRationalPoly. Zero element, or a zero divisor against a
// reducible modulus, -> MathError::division_by_zero.
[[nodiscard]] auto inverse() const -> Result<BigAlgebraicNumber>;
[[nodiscard]] auto divide(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
// Non-negative power by repeated squaring; negative exponent -> MathError::domain_error.
[[nodiscard]] auto pow(std::int64_t exponent) const -> Result<BigAlgebraicNumber>;

// Field norm N(a) = det, trace Tr(a) = trace, of multiplication-by-a on {1,...,alpha^(d-1)}.
[[nodiscard]] auto norm() const -> Result<BigRational>;
[[nodiscard]] auto trace() const -> Result<BigRational>;
```

## Error model

| Condition | Error |
| :--- | :--- |
| `create(minimal)` with a zero polynomial or a nonzero constant | `MathError::domain_error` |
| `add` / `subtract` / `multiply` / `divide` across different fields | `MathError::domain_error` |
| `pow` with a negative exponent | `MathError::domain_error` |
| `inverse()` of the zero element | `MathError::division_by_zero` |
| `inverse()` of a nonzero element that is a zero divisor of a reducible modulus | `MathError::division_by_zero` |
| `divide(o)` where `o` has no inverse | `MathError::division_by_zero` |

There is **no `overflow` row**: `BigRational` / `BigRationalPoly` arithmetic
cannot wrap, so the `overflow` the `int64` [`algnum`](algnum.md) can raise on
saturated residue coefficients is unreachable here.

## Worked example

```cpp
import nimblecas.bigalgnum;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
using namespace nimblecas;

auto q  = [](std::int64_t v) { return BigRational::from_int(v); };
auto qs = [](std::string_view s) { return BigRational::from_string(s).value(); };
auto poly = [](std::vector<std::int64_t> cs) {
    std::vector<BigRational> v;
    for (std::int64_t c : cs) v.push_back(q(c));
    return BigRationalPoly::from_coeffs(std::move(v));
};

// Q(i) = Q[t]/(t^2 + 1): i*i == -1, exactly as in the int64 tier.
const BigNumberField qi = BigNumberField::create(poly({1, 0, 1})).value();
auto alpha = qi.generator().value();
alpha.multiply(alpha).value().is_equal(qi.from_bigrational(q(-1)));  // true

// Q(sqrt2) = Q[t]/(t^2 - 2). With B = 10^10, (B*sqrt2)^2 = 2*B^2 = 2e20 — a residue
// coefficient far beyond int64's ~9.22e18 ceiling, exact only in this tier.
const BigNumberField q2 = BigNumberField::create(poly({-2, 0, 1})).value();
auto g = q2.from_poly(BigRationalPoly::monomial(qs("10000000000"), 1)).value();  // B*sqrt2
auto sq = g.multiply(g).value();
sq.is_equal(q2.from_bigrational(qs("200000000000000000000")));  // (B*sqrt2)^2 == 2e20

// norm(B*sqrt2) == -2*B^2 == -2e20, also past the int64 ceiling.
g.norm().value() == qs("-200000000000000000000");

// Honesty: x^2 - 1 = (x-1)(x+1) is reducible and create() accepts it (no factoring at
// this tier). The residue (x - 1) shares the factor (x-1) with the modulus, so it is a
// zero divisor: inverse() honestly reports division_by_zero, never a fabricated inverse.
auto f = BigNumberField::create(poly({-1, 0, 1})).value();  // x^2 - 1
auto a = f.from_poly(poly({-1, 1})).value();                // x - 1
auto inv = a.inverse();
!inv.has_value() && inv.error() == MathError::division_by_zero;  // true
```

## See also

- [`nimblecas.algnum`](algnum.md) — the `int64`-`RationalPoly`,
  overflow-checked sibling this module big-backs; use it when residue
  coefficients fit `int64`.
- [`nimblecas.bigratpoly`](bigratpoly.md) — the unbounded `Q[x]` residue
  representation.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field elements combine over.
- [Documentation hub](../Index.md)
