# `nimblecas.factor` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/factor/factor.cppm`

Exact **factorization of a polynomial over the rationals `Q[x]` into irreducibles**
(ROADMAP §7.21). Given `p` in `Q[x]`, `factor_over_Q` returns the irreducible
factors of `p` with their multiplicities, using **Yun's square-free
factorization** followed by **Kronecker's algorithm** on each square-free factor.
Everything is exact and finite — no floating point.

```cpp
import nimblecas.factor;
```

Depends on [`core`](core.md), [`polynomial`](polynomial.md) and
[`ratpoly`](ratpoly.md).

## Why this module exists

Neither of the two factorization tools that precede it splits a polynomial into
irreducibles:

- [`rational_roots`](roots.md) recovers only the **linear** factors that
  correspond to rational roots.
- [`square_free_factorization`](polynomial.md) (Yun) separates **repeated**
  factors but leaves each square-free part whole.

The gap is a **reducible, square-free polynomial with no rational roots** — for
example `(x^2 - 2)(x^3 - 2) = x^5 - 2x^3 - 2x^2 + 4`. It has no rational root to
peel and is already square-free, yet it is not irreducible: it is a quadratic
times a cubic, each solvable by radicals. Root-finding needs that split so a
downstream solver can dispatch each factor to its closed form instead of falling
back to numerics. `factor_over_Q` performs it.

## The pipeline

For `p` in `Q[x]`:

1. **Clear denominators** — multiply by the LCM of the coefficient denominators to
   get an integer polynomial (overflow-checked).
2. **Primitive part** — divide by the integer content and normalise to a positive
   leading coefficient.
3. **Square-free factorization** (Yun) — `(square-free primitive factor,
   multiplicity)` pairs, pairwise coprime, whose product of `factor^multiplicity`
   is the primitive part.
4. **Kronecker** — factor each square-free factor into irreducibles.
5. **Combine** multiplicities.

### Kronecker's algorithm

On a square-free primitive integer polynomial `f` of degree `n >= 1`, a proper
factor of degree `<= floor(n/2)` exists **iff** `f` is reducible. For each target
degree `s` from `1` to `floor(n/2)`:

- Pick `s + 1` distinct integer nodes `0, 1, -1, 2, -2, …`.
- If `f(node) = 0`, then `(x - node)` divides `f` — peel that linear factor and
  recurse on the quotient.
- Otherwise every candidate factor `g` satisfies `g(node) | f(node)`. Enumerate
  **one integer divisor `±d` of `f(node)` per node** and **Lagrange-interpolate**
  (exactly, over `Q`) the unique polynomial `g` of degree `<= s` through those
  values. Accept `g` when its coefficients are integral, its degree is `>= 1`, and
  its primitive part **divides `f` exactly**. On acceptance, recurse on the factor
  and its cofactor.
- If no `s` yields a factor, `f` is **irreducible**.

Interpolation is performed by precomputing the cardinal Lagrange basis for the
node set once, then combining it with each divisor tuple; the interpolant is
formed over [`Rational`](ratpoly.md) and only accepted after every coefficient is
verified integral.

## Normalization convention

Each returned factor is a **primitive integer polynomial** (content `1`) with a
**positive leading coefficient**, lifted into `Q[x]`. Writing `prim(·)` for the
primitive part, the product of the returned factors — each raised to its
multiplicity — equals

```
prim( LCM(denominators of p) · p )
```

i.e. `p` up to a **nonzero rational constant**. The rational constant is the
content and leading rational scale that were folded out in steps 1–2; it is not
returned. So `factor_over_Q(2·p)`, `factor_over_Q(p)` and `factor_over_Q(p/3)`
all give the **same** factors. Multiplicities come entirely from the square-free
stage; distinct square-free factors are pairwise coprime, so an irreducible never
appears in more than one of them.

## The honesty boundary

`factor_over_Q` is **exact and complete over `Q`**: for any input it can process
within budget, it returns a *fully* irreducible factorization — never a
plausible-but-wrong or partially-factored result (Rule 32).

The one incompleteness is deliberate and reported honestly. Kronecker's
divisor-tuple search is a Cartesian product over the divisors of the sample
values `f(node)`, which can blow up combinatorially for high-degree polynomials
with large-magnitude samples. The search shares a fixed budget of

```
kDivisorTupleBudget = 1'000'000   // divisor-tuples per factorization
```

Each target-degree search charges the size of its divisor-tuple product to this
budget; if the running total would be exceeded, the call returns
**`MathError::not_implemented`** — an honest "could not factor within budget" —
rather than looping unboundedly or emitting a wrong answer. All exact `int64`
arithmetic is overflow-checked and surfaces **`MathError::overflow`** on a
boundary (such as `INT64_MIN`, whose magnitude is unrepresentable), mirroring the
checked-`gcd` guard in [`ratpoly`](ratpoly.md).

## API

```cpp
[[nodiscard]] auto factor_over_Q(const RationalPoly& p)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>>;
```

| Function | Description |
| :--- | :--- |
| `factor_over_Q(p)` | Irreducible factors of `p` over `Q`, each paired with its multiplicity (`>= 1`), in no particular order. Each factor is primitive, content-normalised with a positive leading coefficient, and irreducible over `Q`. The product of `factor^multiplicity` equals `prim(LCM(denominators)·p)` (see [Normalization](#normalization-convention)). |

## Error model

| Condition | Error |
| :--- | :--- |
| `p` is the zero polynomial (every value is a root) | `MathError::domain_error` |
| A pathological input whose divisor-tuple search would exceed `kDivisorTupleBudget` | `MathError::not_implemented` |
| Clearing denominators, interpolation, or a divisibility test exceeds `int64` (e.g. an `INT64_MIN` boundary) | `MathError::overflow` |

A **nonzero constant** has no non-unit factors and returns an **empty list** — not
an error.

## Worked example

The headline case (integer polynomials given low-degree-first, the `RationalPoly`
convention where `coeffs[i]` is the coefficient of `x^i`):

```cpp
import nimblecas.factor;
import nimblecas.ratpoly;
using namespace nimblecas;

// (x^2 - 2)(x^3 - 2) = x^5 - 2x^3 - 2x^2 + 4      coeffs {4, 0, -2, -2, 0, 1}
factor_over_Q(/* x^5 - 2x^3 - 2x^2 + 4 */).value();
//   -> { (x^2 - 2, 1), (x^3 - 2, 1) }
//      no rational roots, yet it splits into a quadratic and a cubic —
//      each then solvable by radicals.
```

More cases from `tests/factor_tests.cpp`:

```cpp
factor_over_Q(/* x^2 - 1 */).value();        // { (x - 1, 1), (x + 1, 1) }
factor_over_Q(/* x^2 - 2 */).value();        // { (x^2 - 2, 1) }        irreducible
factor_over_Q(/* x^3 - 3x + 2 */).value();   // { (x - 1, 2), (x + 2, 1) }
factor_over_Q(/* x^4 + 1 */).value();        // { (x^4 + 1, 1) }        irreducible
factor_over_Q(/* x^4 - 1 */).value();        // { (x-1,1), (x+1,1), (x^2+1,1) }

factor_over_Q(RationalPoly{}).error();       // MathError::domain_error (zero poly)
factor_over_Q(/* 5 */).value();              // { }  nonzero constant: no factors
```

## See also

- [`nimblecas.polynomial`](polynomial.md) — the integer ring `Z[x]`, `divide_exact`,
  primitive parts, and the Yun `square_free_factorization` this module builds on.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`) over which Lagrange interpolation is carried out.
- [`nimblecas.roots`](roots.md) — rational roots (the linear factors); a fast path
  for the `s = 1` case.
- [`nimblecas.solve`](solve.md) — the consumer: closed-form radical roots, which
  uses this factorization so a reducible degree `>= 5` factor is solved by radicals
  rather than numerically.
- [Documentation hub](../Index.md)
```
