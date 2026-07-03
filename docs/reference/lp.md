# `nimblecas.lp` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/lp/lp.cppm`

**Linear programming done exactly** over the rationals via the Simplex method
(ROADMAP §7.22). `maximize` solves a standard-form linear program

```
maximize  c · x    subject to   A x <= b,   x >= 0,   with every b_i >= 0 given,
```

entirely in exact [`Rational`](ratpoly.md) arithmetic. Because every `b_i >= 0`,
the slack basis (`x = 0`, slacks `= b`) is already feasible, so a **single-phase**
Simplex suffices — there is no Phase I. The point of a rational tableau is that
pivoting is closed under exact arithmetic: unlike a floating-point tableau it
accumulates **no round-off**, so an optimum reported as `p/q` is exactly `p/q`,
and the termination proof (Bland's rule) is never defeated by comparison noise.

```cpp
import nimblecas.lp;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## The method

- **Tableau.** Columns `0 … n-1` are the original variables; columns `n … n+m-1`
  are the slacks, laid out as the identity so row `i` owns slack column `n+i`.
  The initial basis is the slacks, `rhs = b`, and the objective is `0`. Because
  `b >= 0`, this basis is immediately feasible — no Phase I.
- **Entering variable (Bland's rule).** The smallest-index column with a strictly
  positive reduced cost. None ⇒ optimality. Keying on the smallest index is the
  anti-cycling choice that guarantees termination.
- **Leaving variable (min-ratio, Bland tie-break).** The row minimizing
  `rhs[i] / tab[i][entering]` over rows with a **positive** pivot entry, ties
  broken again by the smallest basic-variable index. A column with **no positive
  entry** certifies unboundedness.
- **Exact comparisons.** Every ratio comparison keys off the sign of a numerator
  (denominators are kept positive and canonical by `Rational`), and the
  min-ratio test compares `a/b` against `c/d` by cross-multiplying with the
  positive denominators — `a·d` vs `c·b` — under an overflow check, never
  flipping a sign the way a negative denominator would.

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every pivot divide, eliminate, and objective
update flows through `Rational`'s checked add / subtract / multiply / divide, and
the min-ratio cross-multiplication is guarded, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping.

## API

```cpp
enum class LpStatus : std::uint8_t { optimal, unbounded };

struct LpSolution {
    LpStatus status;
    Rational value;                  // objective c · x at the optimum
    std::vector<Rational> solution;  // one value per original variable
};

[[nodiscard]] auto maximize(const std::vector<std::vector<Rational>>& A,
                            const std::vector<Rational>& b,
                            const std::vector<Rational>& c) -> Result<LpSolution>;
```

### `maximize`

Maximize `c · x` over `{ x : A x <= b, x >= 0 }`, with `A` an `m × n` matrix
(`m` constraints, `n` variables), `b` of length `m` with every `b_i >= 0`, and
`c` of length `n`. The dimensions are derived as `m = A.size()` and
`n = c.size()`.

On success:

- **`LpStatus::optimal`** — `value` is the exact optimal objective and
  `solution` carries one exact value per **original** variable (the internal
  slacks are dropped). A variable that is basic takes the right-hand side of its
  row; a nonbasic variable is `0`.
- **`LpStatus::unbounded`** — the objective can grow without bound. `solution`
  is **empty** and `value` is left at its default `0/1` and is not meaningful.

## Error model

| Condition | Error |
| :--- | :--- |
| `b.size()` disagrees with the number of constraints `m` | `MathError::domain_error` |
| A ragged `A` — some row's width `!= n` (`= c.size()`) | `MathError::domain_error` |
| Any `b_i < 0` (breaks slack-basis feasibility) | `MathError::domain_error` |
| An `int64` numerator or denominator computation in the tableau wraps | `MathError::overflow` |

## Worked examples

From `tests/lp_tests.cpp` (each `A x <= b`, `x >= 0`, maximizing `c · x`):

```cpp
import nimblecas.lp;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// A box: max x + y  s.t.  x <= 4, y <= 3      =>  optimum 7 at (4, 3).
maximize({{ri(1), ri(0)}, {ri(0), ri(1)}}, {ri(4), ri(3)}, {ri(1), ri(1)}).value();
//   status = optimal,  value = 7,   solution = [4, 3]

// A vertex: max 3x + 2y  s.t.  x + y <= 4, x + 3y <= 6   =>  optimum 12 at (4, 0).
maximize({{ri(1), ri(1)}, {ri(1), ri(3)}}, {ri(4), ri(6)}, {ri(3), ri(2)}).value();
//   status = optimal,  value = 12,  solution = [4, 0]

// Fractional pivots, integer vertex: max x + y  s.t.  2x + y <= 3, x + 2y <= 3.
// Intermediate pivots pass through 3/2 values but land exactly on (1, 1).
maximize({{ri(2), ri(1)}, {ri(1), ri(2)}}, {ri(3), ri(3)}, {ri(1), ri(1)}).value();
//   status = optimal,  value = 2,   solution = [1, 1]   (exact)

// A genuinely non-integral vertex: max x + y  s.t.  2x + y <= 4, x + 2y <= 4.
maximize({{ri(2), ri(1)}, {ri(1), ri(2)}}, {ri(4), ri(4)}, {ri(1), ri(1)}).value();
//   status = optimal,  value = 8/3, solution = [4/3, 4/3]   (fractions survive exactly)

// Unbounded: max x  s.t.  x - y <= 1   =>  x grows without bound (raise y too).
maximize({{ri(1), ri(-1)}}, {ri(1)}, {ri(1), ri(0)}).value();
//   status = unbounded,  solution = [] (empty)

// Domain errors.
maximize({{ri(1), ri(0)}, {ri(0), ri(1)}}, {ri(4), ri(-1)}, {ri(1), ri(1)}).error();
//   MathError::domain_error   (negative b entry)
maximize({{ri(1), ri(0)}, {ri(1)}}, {ri(4), ri(3)}, {ri(1), ri(1)}).error();
//   MathError::domain_error   (ragged A)
maximize({{ri(1), ri(0)}, {ri(0), ri(1)}}, {ri(4)}, {ri(1), ri(1)}).error();
//   MathError::domain_error   (b length mismatch)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the tableau
  lives in.
- [`nimblecas.matrix`](matrix.md) — exact dense linear algebra over `Q`
  (determinant, solve, inverse, rank), the sibling exact-`Rational` consumer.
- [`nimblecas.numeric`](numeric.md) — the floating-point half of the numeric
  chain: root-finders on a polynomial.
- [Documentation hub](../Index.md)
