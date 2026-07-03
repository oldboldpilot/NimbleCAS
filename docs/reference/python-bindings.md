# Python bindings (`nimblecas_ext`) — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bindings/bindings.cpp`

A [nanobind](https://github.com/wjakob/nanobind) extension module,
`nimblecas_ext`, exposing the symbolic core to Python. The binding TU is a
non-module translation unit: it `#include`s the nanobind headers and `import`s
the NimbleCAS C++23 modules, which interoperate directly (the `std::string` /
`std::vector` from the headers and from the modules are the same entities).

## Building

The bindings are Linux-only for now (the C++ engine itself is fully
cross-platform). Python dependencies are managed with **uv**.

```bash
scripts/setup_python.sh   # uv sync -> creates .venv, installs nanobind, writes uv.lock
scripts/build.sh          # builds the extension when the uv .venv is present, runs ctest
```

`scripts/build.sh` provisions the uv environment automatically on first use.
The extension is enabled by the CMake option **`-DNIMBLECAS_PYTHON=ON`**.

## Error translation

The C++ core is exception-free (Rule 32, `std::expected`). At the Python
boundary a `MathError` is translated into a Python exception (`nb::value_error`
→ `ValueError`) carrying the `to_string_view` text, which is the idiomatic
Python contract. Functions returning `Result<Expr>` (`rational`, `simplify`,
`differentiate`, `polynomial_gcd`, `square_free_factor`) raise on the error
branch; total functions (`free_of`, `substitute`) never raise.

## `Expr` class

### Static factories

| Python | C++ | Notes |
| :--- | :--- | :--- |
| `Expr.symbol(name)` | `Expr::symbol` | |
| `Expr.integer(value)` | `Expr::integer` | |
| `Expr.real(value)` | `Expr::real` | |
| `Expr.rational(numerator, denominator)` | `Expr::rational` | Raises `ValueError` on a zero denominator (or `INT64_MIN`). |
| `Expr.sum(terms)` | `Expr::sum` | `terms` is a list of `Expr`. |
| `Expr.product(factors)` | `Expr::product` | |
| `Expr.power(base, exponent)` | `Expr::power` | |
| `Expr.apply(name, args)` | `Expr::apply` | |

### Methods and operators

| Python | Behavior |
| :--- | :--- |
| `e.add(other)` / `e + other` | Sum. |
| `e.mul(other)` / `e * other` | Product. |
| `e.pow(exponent)` | Power. |
| `e.is_equivalent_to(other)` | Structural equality. |
| `e.to_string()` / `repr(e)` | Rendering. |
| `e == other` / `e != other` | Structural (in)equality. Returns `NotImplemented` for a non-`Expr` comparand, so `expr == 5` yields `False`/`True` per Python convention instead of raising. |
| `hash(e)` | Structural hash, consistent with `==` (`a == b ⇒ hash(a) == hash(b)`), so `Expr` works correctly in `set` / `dict`. |

### Module functions

| Python | C++ | Notes |
| :--- | :--- | :--- |
| `free_of(u, t)` | `nimblecas::free_of` | `True` if `u` does not contain `t`. |
| `substitute(u, t, r)` | `nimblecas::substitute` | Replace every `t` in `u` with `r`. |
| `simplify(u)` | `nimblecas::simplify` | Raises on a `MathError`. |
| `differentiate(u, var)` | `nimblecas::differentiate` | `var` is a string. |
| `polynomial_gcd(a, b, var)` | `nimblecas::polynomial_gcd` | |
| `square_free_factor(u, var)` | `nimblecas::square_free_factor` | Returns a list of `(Expr, int)` `(factor, multiplicity)` pairs. |

## Example

The smoke test `tests/test_bindings.py` is the usage source of truth:

```python
import nimblecas_ext as ncas

x = ncas.Expr.symbol("x")
y = ncas.Expr.symbol("y")

assert x == ncas.Expr.symbol("x")
assert x != y

# u = x^2 + x
u = x.pow(ncas.Expr.integer(2)).add(x)
assert not ncas.free_of(u, x)
assert ncas.free_of(u, ncas.Expr.symbol("z"))

# substitute x -> y  =>  y^2 + y (u unchanged, immutable)
r = ncas.substitute(u, x, y)
assert r.is_equivalent_to(y.pow(ncas.Expr.integer(2)).add(y))

# operator sugar
assert (x + y).is_equivalent_to(ncas.Expr.sum([x, y]))
assert (x * y).is_equivalent_to(ncas.Expr.product([x, y]))

# comparison with a non-Expr does not raise
assert (x == 5) is False

# __hash__ dedupes in a set
assert len({x, ncas.Expr.symbol("x"), y}) == 2

# simplify and differentiate
assert ncas.simplify(x + x).is_equivalent_to(ncas.Expr.integer(2).mul(x))
d = ncas.differentiate(x.pow(ncas.Expr.integer(2)), "x")   # 2*x

# polynomial gcd and square-free factorization
g = ncas.polynomial_gcd(u.add(ncas.Expr.integer(-1)), x, "x")
factors = ncas.square_free_factor(
    x.add(ncas.Expr.integer(1)).pow(ncas.Expr.integer(2)), "x")
assert len(factors) == 1 and factors[0][1] == 2

# error translation: zero denominator -> ValueError
try:
    ncas.Expr.rational(1, 0)
except ValueError as exc:
    assert "division by zero" in str(exc)
```

## See also

- [`nimblecas.symbolic`](symbolic.md), [`simplify`](simplify.md), [`diff`](diff.md), [`polyexpr`](polyexpr.md) — the exposed C++ API.
- [Quickstart — Python environment](../QUICKSTART.md)
- [Documentation hub](../Index.md)
