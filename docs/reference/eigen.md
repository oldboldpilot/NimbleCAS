# `nimblecas.eigen` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/eigen/eigen.cppm`

Exact spectral primitives over the rationals (ROADMAP §7.10). This module answers
three linear-algebra questions about a square [`Matrix`](matrix.md) of exact
[`Rational`](ratpoly.md) entries **without ever touching floating point**: the
**characteristic polynomial** `p(λ) = det(λI − A)`, the **rational eigenvalues**
(those eigenvalues that happen to lie in `Q`, with multiplicity), and a **basis
for the eigenspace** of a given rational eigenvalue. Every computation stays
inside exact rational arithmetic — the char-poly via the Faddeev–LeVerrier
recurrence (matrix multiply / scale / trace / add over `Rational`, no symbolic
determinant), the eigenvalues via the rational-root theorem in
[`roots`](roots.md), and the eigenvectors via exact `Rational` row reduction
(RREF).

The **honest boundary** is scope, not precision. `rational_eigenvalues` returns
**only the rational slice** of the spectrum: irrational eigenvalues (e.g. the
`±√2` of `[[0,2],[1,0]]`) and complex eigenvalues (e.g. the `±i` of a rotation)
are deliberately **not** returned, because representing them exactly needs
algebraic-number support that is out of scope here. This is an exact partial
answer, never a numerical approximation — what is returned is provably correct,
and what is omitted is omitted, not rounded. Following the rest of the engine,
arithmetic is **overflow-checked** (Rule 32): any `int64` numerator or
denominator that would wrap surfaces as `MathError::overflow` rather than
silently corrupting a result.

```cpp
import nimblecas.eigen;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`roots`](roots.md).

## Free functions

All three entry points are free functions in `namespace nimblecas`, each
`[[nodiscard]]` and returning a `Result<T>`. All require a **square** matrix and
reject a non-square one with `MathError::domain_error`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `characteristic_polynomial` | `auto characteristic_polynomial(const Matrix& a) -> Result<RationalPoly>` | The monic polynomial `p(λ) = det(λI − A)` of degree `n`, computed by the Faddeev–LeVerrier recurrence over `Rational`. Coefficients are ascending: `coefficient(i)` is the coefficient of `λ^i`, with `coefficient(n) == 1` (monic). The `0×0` matrix yields the constant polynomial `1` (empty-product convention). Propagates any overflow from the underlying matrix arithmetic. |
| `rational_eigenvalues` | `auto rational_eigenvalues(const Matrix& a) -> Result<std::vector<std::pair<Rational, std::int64_t>>>` | The **rational** eigenvalues of `A`: the rational roots of the characteristic polynomial, each paired with its algebraic multiplicity `(root, multiplicity)`. Irrational and complex eigenvalues are **not** included (out of scope). Roots come back in no particular order. |
| `eigenvectors_for` | `auto eigenvectors_for(const Matrix& a, const Rational& eigenvalue) -> Result<std::vector<std::vector<Rational>>>` | A basis for the eigenspace of `eigenvalue`: a basis for the null space (kernel) of `(A − eigenvalue·I)`, obtained by exact `Rational` RREF. One basis vector per free column; each inner vector has length `n`. If `eigenvalue` is **not** an eigenvalue (trivial kernel) the result is **empty**, not an error. |

### Notes on the contract

- **Ascending coefficient convention.** `characteristic_polynomial` returns a
  monic polynomial with `coefficient(i)` the coefficient of `λ^i`. For a 2×2 the
  familiar identities hold: `coefficient(1) == −trace(A)` and
  `coefficient(0) == det(A)`.
- **Multiplicity is *algebraic*.** The `std::int64_t` paired with each eigenvalue
  is the multiplicity of that root in `p(λ)` (the algebraic multiplicity), which
  can exceed the geometric multiplicity — the number of basis vectors
  `eigenvectors_for` returns. A Jordan block `[[2,1],[0,2]]` reports eigenvalue
  `2` with algebraic multiplicity `2`, yet its eigenspace is only
  1-dimensional.
- **`eigenvectors_for` does not require a *rational* eigenvalue in the spectrum
  to succeed.** It simply computes `ker(A − λI)` for whatever `Rational` λ you
  pass. If λ is not an eigenvalue, the kernel is trivial and you get an empty
  basis — that empty vector is the honest "λ is not in the spectrum" signal.
- **Empty matrix.** For `n == 0`, `characteristic_polynomial` returns the
  constant `1` and `eigenvectors_for` returns an empty basis.

## Error model

| Condition | Error |
| :--- | :--- |
| `a` is not square (any of the three functions) | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps during the Faddeev–LeVerrier recurrence, root search, or row reduction | `MathError::overflow` |
| Any other error raised by the underlying `Matrix` / `RationalPoly` / `roots` arithmetic | propagated unchanged |

`rational_eigenvalues` runs `characteristic_polynomial` first and forwards its
`domain_error` (non-square) or `overflow` unchanged. Because the characteristic
polynomial is always monic — hence never the zero polynomial —
`rational_eigenvalues` never triggers the `domain_error` that
[`rational_roots`](roots.md) raises for the zero polynomial. A matrix with **no**
rational eigenvalues (e.g. one whose eigenvalues are all irrational or complex)
is **not** an error: it returns an **empty** vector.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;
using namespace nimblecas;

auto r = [](std::int64_t v) { return Rational::from_int(v); };

// diag(2, 3): p(λ) = (λ − 2)(λ − 3) = λ² − 5λ + 6, monic.
const auto a = Matrix::from_rows({{r(2), r(0)}, {r(0), r(3)}}).value();

const auto p = characteristic_polynomial(a).value();
p.degree();          // 2
p.coefficient(0);    // 6   (= det A)
p.coefficient(1);    // −5  (= −trace A)
p.coefficient(2);    // 1   (monic leading term)

// Rational eigenvalues: {2, 3}, each with algebraic multiplicity 1.
const auto evs = rational_eigenvalues(a).value();
// evs contains (2, 1) and (3, 1) in some order.

// Eigenspace of 2 is the x-axis; a null-space basis vector is unique only up to
// scaling, so verify the defining property A·v = λ·v rather than a literal value.
const auto v2 = eigenvectors_for(a, r(2)).value();
v2.size();           // 1   (1-dimensional eigenspace)
v2[0].size();        // 2   (each vector has length n)
v2[0][1] == r(0);    // true — lives on the x-axis

// 4 is not an eigenvalue of diag(2, 3): (A − 4I) is invertible, so the kernel
// is trivial and NO basis vectors come back — an empty result, not an error.
eigenvectors_for(a, r(4)).value().empty();   // true

// A Jordan block: algebraic multiplicity 2, but a 1-dimensional eigenspace.
const auto j = Matrix::from_rows({{r(2), r(1)}, {r(0), r(2)}}).value();
const auto jev = rational_eigenvalues(j).value();
jev.size();          // 1   — single distinct eigenvalue
jev[0].first == r(2);        // true
jev[0].second;               // 2   (algebraic multiplicity)
eigenvectors_for(j, r(2)).value().size();   // 1 (geometric multiplicity < 2)

// A repeated eigenvalue with a genuinely 2-dimensional eigenspace.
const auto d = Matrix::from_rows({{r(3), r(0), r(0)},
                                  {r(0), r(5), r(0)},
                                  {r(0), r(0), r(5)}}).value();
eigenvectors_for(d, r(5)).value().size();   // 2 (eigenvalue 5 is 2-dimensional)

// Non-square input is a domain error for the whole module.
const auto wide = Matrix::from_rows({{r(1), r(2), r(3)},
                                     {r(4), r(5), r(6)}}).value();
characteristic_polynomial(wide).error();    // MathError::domain_error
rational_eigenvalues(wide).error();          // MathError::domain_error
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrix whose spectrum
  this module analyzes.
- [`nimblecas.roots`](roots.md) — the rational-root theorem solver that extracts
  the rational eigenvalues from the characteristic polynomial.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and the
  `RationalPoly` the characteristic polynomial lives in.
- [`nimblecas.complex`](complex.md) — where the complex eigenvalues this module
  omits would eventually be represented.
- [Documentation hub](../Index.md)
