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

## Financial submodules (`pricing`, `finance`, `analytics`)

Beyond the symbolic core, `nimblecas_ext` exposes thin wrappers over the
financial-mathematics engine. **These reuse the existing CAS engine — the Python
layer computes nothing itself; it forwards to `pricing::…` / `finance::…` /
`analytics::…` and translates the `Result<T>` error branch to `ValueError`.** In
particular `pricing.monte_carlo` calls the same `pricing::monte_carlo_european`
(the reproducible counter-RNG engine) that the CUDA and Triton kernels accelerate
and validate against — there is one Monte-Carlo implementation, not three.

| Python | C++ engine reused |
| :--- | :--- |
| `pricing.black_scholes(spot, strike, rate, div, vol, expiry, is_call=True)` | `pricing::black_scholes_price` |
| `pricing.greeks(...)` | `pricing::black_scholes_greeks` |
| `pricing.implied_volatility(...)` | `pricing::implied_volatility` |
| `pricing.monte_carlo(..., paths=1e6, seed=42) -> (price, std_error)` | `pricing::monte_carlo_european` |
| `pricing.trinomial(...)` | `pricing::trinomial_price` |
| `finance.irr(values, guess=0.1)`, `finance.mirr`, `finance.rate`, `finance.nper` | `finance::irr` / `mirr` / `rate` / `nper` |
| `analytics.sharpe_ratio(returns, rf=0, periods_per_year=1)` | `analytics::sharpe_ratio` |
| `analytics.max_drawdown`, `analytics.value_at_risk` | `analytics::…` |
| `analytics.min_variance_weights(cov)`, `analytics.tangency_weights(...)` | `analytics::…` |

Verified through the built extension on clang++-22 (nanobind 2.13):

```python
import nimblecas_ext as n
assert abs(n.pricing.black_scholes(100, 100, 0.05, 0.0, 0.2, 1.0, True) - 10.4506) < 1e-3
assert abs(n.finance.irr([-100.0, 110.0]) - 0.10) < 1e-9
w = n.analytics.min_variance_weights([[0.04, 0.0], [0.0, 0.09]])   # -> [0.6923, 0.3077]
price, se = n.pricing.monte_carlo(100, 100, 0.05, 0.0, 0.2, 1.0, True, 1_000_000, 42)  # CAS CPU MC
```

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
