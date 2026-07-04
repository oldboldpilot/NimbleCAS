# `nimblecas.interpolation` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/interpolation/interpolation.cppm`

Exact polynomial interpolation over the **rationals** `Q` (ROADMAP §7). Given
distinct nodes `x_0,...,x_{n-1}` and values `y_0,...,y_{n-1}` in `Q`, there is a
**unique** polynomial of degree `<= n-1` through the points `(x_i, y_i)`; this
module builds and evaluates that polynomial **exactly** — **no floating point
ever enters** — through four classical, mathematically equivalent constructions:
**Lagrange** (basis products), **Newton** (divided differences, with cheap
incremental point addition), **barycentric** (weights + stable quotient), and
**Neville** (tableau evaluation without forming the polynomial). Because the same
unique polynomial underlies all four, they agree exactly. **Hermite**
interpolation additionally matches first derivatives, returning the unique degree
`<= 2n-1` polynomial with `p(x_i) = y_i` and `p'(x_i) = y'_i`.

**Honesty boundary.** For rational data this interpolant is the exact unique
polynomial — there is *no approximation error to report*. The Runge phenomenon
and ill-conditioning are properties of the **choice of nodes** (equispaced
high-degree nodes make the true interpolant oscillate wildly between them), *not*
numerical errors introduced here: the coefficients returned are the
mathematically exact ones. The only failures are structural (bad input) or an
`int64` overflow inside an exact `Rational` step.

```cpp
import nimblecas.interpolation;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md). All symbols live in
namespace `nimblecas`.

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every step flows through `Rational`'s checked
add / subtract / multiply / divide / negate, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping. There is no rounding and no conditioning loss to hide — an
overflow is the *only* arithmetic failure mode, and it is reported, never
absorbed.

## Structural preconditions

Every `(nodes, values)` interpolator shares one precondition: the two spans must
be **equal in length**, **non-empty**, and the nodes **pairwise distinct**.
Duplicate nodes make interpolation ill-posed (a zero denominator `x_i - x_j`), so
they are rejected rather than silently divided. Any violation returns
`MathError::domain_error`. Hermite additionally requires `derivatives` to match
the node count.

## Free functions

### Exact evaluation

```cpp
[[nodiscard]] auto poly_evaluate(const RationalPoly& p, const Rational& x)
    -> Result<Rational>;
```

| Function | Behavior |
| :--- | :--- |
| `poly_evaluate` | `p(x)` evaluated exactly at a rational point via Horner. Handy for checking that an interpolant passes through its nodes; fails only on `overflow`. |

### Lagrange

```cpp
[[nodiscard]] auto lagrange_basis(std::span<const Rational> nodes, std::size_t i)
    -> Result<RationalPoly>;
[[nodiscard]] auto lagrange_basis_polynomials(std::span<const Rational> nodes)
    -> Result<std::vector<RationalPoly>>;
[[nodiscard]] auto lagrange_polynomial(std::span<const Rational> nodes,
                                       std::span<const Rational> values)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `lagrange_basis` | The `i`-th basis polynomial `L_i(x) = Π_{j!=i} (x - x_j)/(x_i - x_j)`: the unique degree `<= n-1` polynomial with `L_i(x_i) = 1` and `L_i(x_j) = 0` for `j != i`. Fails `domain_error` on an empty node set, an out-of-range index `i >= n`, or duplicate nodes. |
| `lagrange_basis_polynomials` | All `n` basis polynomials `L_0,...,L_{n-1}` for the nodes. Fails `domain_error` on empty or duplicate nodes. |
| `lagrange_polynomial` | The interpolant `Σ_i y_i L_i(x)`: the unique degree `<= n-1` polynomial through the points. Fails `domain_error` on size mismatch, empty input, or duplicate nodes. |

### Newton divided differences

```cpp
[[nodiscard]] auto divided_differences(std::span<const Rational> nodes,
                                       std::span<const Rational> values)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto newton_polynomial(std::span<const Rational> nodes,
                                     std::span<const Rational> values)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `divided_differences` | The Newton coefficients `c_k = f[x_0,...,x_k]` (the top diagonal of the divided-difference table), so the interpolant is `c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...`. Fails `domain_error` on size mismatch, empty input, or duplicate nodes. |
| `newton_polynomial` | The interpolant assembled in Newton form — the **identical polynomial** to `lagrange_polynomial`. Same preconditions. |

### Barycentric weights

```cpp
[[nodiscard]] auto barycentric_weights(std::span<const Rational> nodes)
    -> Result<std::vector<Rational>>;
```

| Function | Behavior |
| :--- | :--- |
| `barycentric_weights` | The weights `w_i = 1 / Π_{j!=i} (x_i - x_j)`. Fails `domain_error` on empty or duplicate nodes. |

### Neville

```cpp
[[nodiscard]] auto neville_evaluate(std::span<const Rational> nodes,
                                    std::span<const Rational> values,
                                    const Rational& x)
    -> Result<Rational>;
```

| Function | Behavior |
| :--- | :--- |
| `neville_evaluate` | Evaluate the interpolant at `x` by recombining the tableau of lower-order interpolants — **the polynomial is never formed**. Exact over `Q`; agrees with every other method. Same `(nodes, values)` preconditions. |

### Hermite (value + first derivative)

```cpp
[[nodiscard]] auto hermite_polynomial(std::span<const Rational> nodes,
                                      std::span<const Rational> values,
                                      std::span<const Rational> derivatives)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `hermite_polynomial` | The unique degree `<= 2n-1` polynomial with `p(x_i) = values[i]` and `p'(x_i) = derivatives[i]` for every node, built from confluent (repeated-node) divided differences. Fails `domain_error` when the three spans differ in length, the input is empty, or a node repeats. |

## `NewtonInterpolant` — incremental Newton form

An immutable/fluent Newton interpolant that supports **cheap incremental point
addition**: adding one node costs `O(current size)` exact rational steps rather
than rebuilding the whole table. `with_point` returns a *new* interpolant rather
than mutating in place (railway style).

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `NewtonInterpolant()` | The empty interpolant — the zero polynomial. |
| `from_points` | `static auto from_points(std::span<const Rational> nodes, std::span<const Rational> values) -> Result<NewtonInterpolant>` | Build directly from a full point set. Fails `domain_error` on mismatch / empty / duplicate. |
| `with_point` | `auto with_point(const Rational& x, const Rational& y) const -> Result<NewtonInterpolant>` | Return a copy extended by one more point; the new Newton coefficient is folded in from the existing ones in `O(size)`. Fails `domain_error` if `x` duplicates an existing node, or `overflow`. |

### Accessors and evaluation

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_empty` | `auto is_empty() const noexcept -> bool` | `true` when no points have been added. |
| `size` | `auto size() const noexcept -> std::size_t` | Number of interpolation points. |
| `nodes` | `auto nodes() const noexcept -> std::span<const Rational>` | The nodes `x_0,...,x_{k-1}`. |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const Rational>` | The Newton coefficients `c_k = f[x_0,...,x_k]`. |
| `polynomial` | `auto polynomial() const -> Result<RationalPoly>` | The interpolant as an explicit polynomial in `Q[x]`. |
| `evaluate` | `auto evaluate(const Rational& x) const -> Result<Rational>` | Exact evaluation via nested Newton (Horner) form. The empty interpolant returns exact `0`. |

## `BarycentricInterpolant` — exact barycentric form

Precomputes the barycentric weights once, then evaluates
`p(x) = [Σ_i w_i/(x - x_i)·y_i] / [Σ_i w_i/(x - x_i)]` exactly. When `x`
coincides with a node the quotient is `0/0`, so the stored value `y_i` is
returned directly.

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| `make` | `static auto make(std::span<const Rational> nodes, std::span<const Rational> values) -> Result<BarycentricInterpolant>` | Build from a point set (weights precomputed). Fails `domain_error` on mismatch / empty / duplicate. |

The default constructor is private; construct via `make`.

### Accessors and evaluation

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `nodes` | `auto nodes() const noexcept -> std::span<const Rational>` | The nodes. |
| `values` | `auto values() const noexcept -> std::span<const Rational>` | The values `y_i`. |
| `weights` | `auto weights() const noexcept -> std::span<const Rational>` | The barycentric weights `w_i`. |
| `evaluate` | `auto evaluate(const Rational& x) const -> Result<Rational>` | Exact barycentric evaluation; returns `y_i` directly at a node, else the exact quotient. Fails only on `overflow`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `nodes.size() != values.size()` (or `!= derivatives.size()` for Hermite) | `MathError::domain_error` |
| Empty input | `MathError::domain_error` |
| A duplicated node (also `with_point` re-adding an existing node) | `MathError::domain_error` |
| `lagrange_basis` index `i` out of range (`i >= nodes.size()`) | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

There is **no interpolation-error term**: for rational data the result is the
exact unique polynomial. Node-choice pathologies (Runge oscillation,
ill-conditioning) are mathematical properties of the sample points, not failures
this layer reports — the returned coefficients are exact. Accessors (`size`,
`nodes`, `weights`, ...) are total and never error.

## Worked examples

```cpp
import nimblecas.interpolation;
import nimblecas.ratpoly;
import nimblecas.polynomial;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto rats = [](std::vector<std::int64_t> xs) {
    std::vector<Rational> r;
    for (std::int64_t v : xs) r.push_back(Rational::from_int(v));
    return r;
};
// A RationalPoly from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// --- Recover a known polynomial exactly ---
// x^2 sampled at 0,1,2 -> the unique degree<=2 interpolant is exactly x^2.
auto nodes  = rats({0, 1, 2});
auto values = rats({0, 1, 4});
lagrange_polynomial(nodes, values).value().is_equal(ipoly({0, 0, 1}));  // true
newton_polynomial(nodes, values).value().is_equal(ipoly({0, 0, 1}));    // true (same poly)

// --- All four methods agree ---
// x^2 + x + 1 on 0..3.
auto xs = rats({0, 1, 2, 3});
auto ys = rats({1, 3, 7, 13});
auto lag  = lagrange_polynomial(xs, ys).value();
auto bary = BarycentricInterpolant::make(xs, ys).value();
const Rational probe = rat(7, 3);                       // an off-node point
auto pv = poly_evaluate(lag, probe).value();
bary.evaluate(probe).value() == pv;                     // true
neville_evaluate(xs, ys, probe).value() == pv;          // true

// --- Newton incremental: point-by-point == all-at-once ---
auto n2 = rats({0, 1, 2, 3});
auto v2 = rats({1, 2, 5, 10});                          // x^2 + 1
auto batch = NewtonInterpolant::from_points(n2, v2).value();
NewtonInterpolant inc;                                  // empty (zero polynomial)
for (std::size_t i = 0; i < n2.size(); ++i)
    inc = inc.with_point(n2[i], v2[i]).value();
inc.size();                                             // 4
inc.polynomial().value().is_equal(batch.polynomial().value());  // true
inc.polynomial().value().is_equal(ipoly({1, 0, 1}));    // true: x^2 + 1
inc.with_point(ri(1), ri(99)).error();                  // domain_error (duplicate node)

// --- Fractional data stays exact (no floats) ---
// Line through (1/2, 1/3) and (3/2, 5); midpoint x = 1 gives exactly 8/3.
std::vector<Rational> fn{rat(1, 2), rat(3, 2)};
std::vector<Rational> fv{rat(1, 3), ri(5)};
auto fl = lagrange_polynomial(fn, fv).value();
poly_evaluate(fl, rat(1, 1)).value() == rat(8, 3);      // true

// --- Hermite: match value AND first derivative ---
// (0, 0, slope 1), (1, 1, slope 2).
std::vector<Rational> hn{ri(0), ri(1)};
std::vector<Rational> hv{ri(0), ri(1)};
std::vector<Rational> hd{ri(1), ri(2)};
auto h  = hermite_polynomial(hn, hv, hd).value();
auto hp = h.derivative().value();
poly_evaluate(h,  ri(0)).value() == ri(0);              // value matched
poly_evaluate(hp, ri(1)).value() == ri(2);              // slope matched
h.degree() <= 3;                                        // degree <= 2n-1 = 3

// One-point Hermite (a=2, v=5, slope=3) is the exact line 3x - 1.
hermite_polynomial(rats({2}), rats({5}), rats({3})).value()
    .is_equal(ipoly({-1, 3}));                          // true

// --- Error handling ---
auto dupN = rats({0, 1, 1});
lagrange_polynomial(dupN, rats({0, 1, 2})).error();     // domain_error (duplicate)
lagrange_polynomial(rats({0, 1, 2}), rats({0, 1})).error();  // domain_error (mismatch)
std::vector<Rational> empty;
lagrange_polynomial(empty, empty).error();              // domain_error (empty)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and the
  `RationalPoly` these interpolants are built from.
- [`nimblecas.polynomial`](polynomial.md) — the integer ring `Z[x]` that
  `RationalPoly` lifts from.
- [`nimblecas.pade`](pade.md) — exact *rational* approximation, the sibling that
  fits `p(x)/q(x)` rather than a single polynomial.
- [`nimblecas.orthopoly`](orthopoly.md), [`nimblecas.combinatorics`](combinatorics.md) —
  other exact `ratpoly`-consuming numeric modules.
- [Documentation hub](../Index.md)
