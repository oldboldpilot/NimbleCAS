# `nimblecas.polyexpr` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/polyexpr/polyexpr.cppm`

The bridge between the symbolic [`Expr`](symbolic.md) model and the dense
[`Polynomial`](polynomial.md) model. It extracts a univariate, integer-
coefficient polynomial in a chosen variable from an expression, rebuilds an
expression from a polynomial, and exposes polynomial GCD and square-free
factorization at the `Expr` level.

```cpp
import nimblecas.polyexpr;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), and
[`polynomial`](polynomial.md).

## API

```cpp
[[nodiscard]] auto to_polynomial(const Expr& u, std::string_view var) -> Result<Polynomial>;
[[nodiscard]] auto from_polynomial(const Polynomial& p, std::string_view var) -> Expr;
[[nodiscard]] auto polynomial_gcd(const Expr& a, const Expr& b, std::string_view var)
    -> Result<Expr>;
[[nodiscard]] auto square_free_factor(const Expr& u, std::string_view var)
    -> Result<std::vector<std::pair<Expr, std::int64_t>>>;
```

### `to_polynomial(u, var)`

Extracts a univariate integer-coefficient `Polynomial` in `var` from `u`,
recursing through sums (accumulate-add), products (accumulate-multiply), and
integer powers (repeated multiply). Fails with **`MathError::not_implemented`**
when `u` is not such a polynomial:

| Rejected input | Reason |
| :--- | :--- |
| a symbol other than `var` | not univariate in `var` |
| a non-integer constant (double or rational) | coefficients must be exact integers |
| a power with a non-integer, non-constant, or negative exponent | not a polynomial term |
| any function application (e.g. `sin(x)`) | not a polynomial |

(Overflow in the underlying polynomial arithmetic surfaces as
`MathError::overflow`.)

### `from_polynomial(p, var)`

Rebuilds a sum-of-monomials `Expr` in the symbol `var` from `p`. Zero
coefficients are omitted; the `x^0` term is a bare constant, the `x^1` term is
`c * x`, and higher terms are `c * x^i`. The zero polynomial becomes
`integer(0)`. The result is **not** simplified — run [`simplify`](simplify.md)
if you need a canonical form.

### `polynomial_gcd(a, b, var)`

Converts both `a` and `b` to polynomials in `var`, computes their
[`Polynomial::gcd`](polynomial.md), and returns the result as an `Expr` via
`from_polynomial`. Propagates any conversion or arithmetic `MathError`.

### `square_free_factor(u, var)`

Converts `u` to a polynomial in `var`, runs
[`Polynomial::square_free_factorization`](polynomial.md), and maps each
`(Polynomial, multiplicity)` pair back to an `(Expr, multiplicity)` pair.

## Example

```cpp
import nimblecas.polyexpr;
import nimblecas.simplify;
using namespace nimblecas;

const Expr x = Expr::symbol("x");

// gcd(x^2 - 1, x - 1) = x - 1
const Expr x2m1 = x.pow(Expr::integer(2)).add(Expr::integer(-1));
const Expr xm1  = x.add(Expr::integer(-1));
auto g = polynomial_gcd(x2m1, xm1, "x");           // x - 1

// square-free factorization of (x + 1)^2 -> one factor, multiplicity 2
auto factors = square_free_factor(
    x.add(Expr::integer(1)).pow(Expr::integer(2)), "x");
assert(factors && factors->size() == 1 && (*factors)[0].second == 2);

// rejection: a function is not a polynomial
auto bad = to_polynomial(Expr::apply("sin", {x}), "x");  // MathError::not_implemented
assert(!bad);
```

## See also

- [`nimblecas.polynomial`](polynomial.md) — the dense polynomial engine.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model.
- [Documentation hub](../Index.md)
