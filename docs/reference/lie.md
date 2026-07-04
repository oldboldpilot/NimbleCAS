# `nimblecas.lie` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/lie/lie.cppm`

Matrix Lie algebras and Lie transforms done **exactly** over the rationals
(ROADMAP §7.x). A matrix Lie algebra is a vector space of square matrices over
`Q` closed under the **Lie bracket** `[A,B] = AB − BA`. This module is the
exact-over-`Q` toolkit for such algebras: the bracket itself, the **structure
constants** `c^k_ij` of a chosen basis (found by an exact rational linear solve
and checked for closure), the **adjoint representation** `ad_X` and the
**Killing form** `K(X,Y) = trace(ad_X ad_Y)`, the **exponential map** (delegated
to [`matexp`](matexp.md)), and **truncated Lie series** / adjoint-action
transforms. Everything that can be exact **is** exact — the bracket, structure
constants, adjoint matrices, and Killing form carry no rounding whatsoever. The
two honestly-approximate corners are named as such: `exponential_map` inherits
[`matexp`](matexp.md)'s contract (exact `e^A` **iff** `A` is nilpotent, else a
rational Taylor approximation of a generally transcendental value), and the Lie
series is a **truncated** partial sum that equals the full transform only in the
limit `order → ∞`, or once `order` reaches the nilpotency index of `ad_L`.

```cpp
import nimblecas.lie;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (`Rational`),
[`matrix`](matrix.md) (exact dense linear algebra over `Q`), and
[`matexp`](matexp.md) (the exponential map).

## The exact-vs-truncation boundary

Everything rides the `Result` railway (Rule 32): dimension violations and a
basis that is **not closed** under the bracket surface as
`MathError::domain_error`, and any `Rational` `int64` overflow is propagated,
never silently wrapped. Within that railway the module draws a sharp honesty
line:

- **Exact over `Q` (no approximation).** `lie_bracket`, `structure_constants`,
  `adjoint_matrix`, and `killing_form` are exact. Bilinearity, antisymmetry
  `[A,B] = −[B,A]`, and the Jacobi identity hold **identically**. The structure
  constants are the *unique* `c^k_ij` with `[X_i,X_j] = Σ_k c^k_ij X_k`,
  recovered by an exact rational solve; if a bracket leaves the span the algebra
  is not closed and the call is `domain_error`.
- **Truncated but exact-per-term.** `lie_series` / `adjoint_action_series`
  compute the partial sum `Σ_{k=0}^{order} (t^k/k!) ad_L^k(f)` with exact
  iterated brackets. Every term kept is exact; the sum equals the whole
  (infinite) transform **only** in the limit `order → ∞`, or — for a nilpotent
  `ad_L` — once `order` reaches the nilpotency index (the omitted tail is then
  identically zero). Otherwise it is an honest exact **partial** sum, never a
  claim of the full `Ad_{exp(L)}`.
- **Inherited (exact-iff-nilpotent).** `exponential_map` forwards to
  [`matexp`](matexp.md)'s `matrix_exp_taylor`: the true `e^A` is generally
  transcendental and **not** representable over `Q`. The returned truncated
  Taylor polynomial equals `e^A` exactly **iff** `A` is nilpotent and `terms` is
  at least `A`'s nilpotency index; for any other `A` it is an exact rational
  approximation.

## Free functions

All symbols are in namespace `nimblecas`. Every function is `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `lie_bracket` | `auto lie_bracket(const Matrix& a, const Matrix& b) -> Result<Matrix>` | The Lie bracket `[A,B] = AB − BA`, exact over `Q`. Also the adjoint action `ad_A(B) = [A,B]`. `A` and `B` must be square and of equal order, else `domain_error`; entry overflow is propagated. |
| `structure_constants` | `auto structure_constants(const std::vector<Matrix>& basis) -> Result<StructureConstants>` | The tensor `c^k_ij` with `[X_i,X_j] = Σ_k c^k_ij X_k` for a basis `{X_0..X_{n-1}}` of square matrices. The basis must be linearly independent (else the solve is singular → `domain_error`) and **closed** under the bracket (else `domain_error`). Overflow is propagated. |
| `adjoint_matrix` | `auto adjoint_matrix(const Matrix& x, const std::vector<Matrix>& basis) -> Result<Matrix>` | `ad_X` as an exact `n × n` rational matrix in `basis`: entry `(k,j)` is the coefficient of `X_k` in `[X, X_j]`, so `(ad_X)(coords of Z) = coords of [X,Z]`. Requires a linearly independent basis closed under bracketing with `X`, and `X` sharing the basis order; else `domain_error`. Overflow propagated. |
| `killing_form` | `auto killing_form(const Matrix& x, const Matrix& y, const std::vector<Matrix>& basis) -> Result<Rational>` | The Killing form `K(X,Y) = trace(ad_X ad_Y)`, an exact `Rational` from the adjoint matrices. Same `domain_error` / overflow conditions as `adjoint_matrix`. |
| `exponential_map` | `auto exponential_map(const Matrix& a, std::int64_t terms) -> Result<Matrix>` | `exp(A)` as the truncated Taylor polynomial `Σ_{k=0}^{terms-1} A^k/k!`, forwarded to [`matexp`](matexp.md). **Exact** `e^A` iff `A` is nilpotent and `terms ≥ A`'s nilpotency index; otherwise an exact rational **approximation** of the generally transcendental `e^A`. Requires `A` square and `terms ≥ 1`, else `domain_error`. |
| `lie_series` | `auto lie_series(const Matrix& l, const Matrix& f, const Rational& t, std::int64_t order) -> Result<Matrix>` | The truncated Lie series `exp(t·ad_L) f = Σ_{k=0}^{order} (t^k/k!) ad_L^k(f)`, where `ad_L^k(f)` is the `k`-fold nested bracket `[L,[L,…[L,f]…]]`. Each term is exact; the sum is truncated at `order` (see the honesty boundary above). `L` and `f` must be square and of equal order, and `order ≥ 0`, else `domain_error`. Overflow propagated. |
| `adjoint_action_series` | `auto adjoint_action_series(const Matrix& l, const Matrix& f, std::int64_t order) -> Result<Matrix>` | The adjoint action `Ad_{exp(L)} f = exp(L) f exp(−L)` approximated by its Lie series at `t = 1`: `Σ_{k=0}^{order} ad_L^k(f)/k!`. A convenience wrapper over `lie_series(l, f, 1, order)` with the identical truncation honesty. |

## `StructureConstants` — the tensor `c^k_ij` of a basis

Returned by `structure_constants`. `c^k_ij` is the coefficient of `X_k` in the
expansion of `[X_i, X_j]`; antisymmetry `c^k_ij = −c^k_ji` holds by construction
of the bracket. The constants are stored as a flat `n³` vector at index
`((i·n) + j)·n + k`.

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| default | `StructureConstants()` | Empty (dimension `0`). |
| construct | `StructureConstants(std::size_t dim, std::vector<Rational> data)` | Wrap a dimension `n` and its flat `n³` coefficient vector. |
| `dimension` | `auto dimension() const noexcept -> std::size_t` | The number of basis elements `n`; indices `i,j,k` run in `[0,n)`. |
| `at` | `auto at(std::size_t i, std::size_t j, std::size_t k) const -> const Rational&` | `c^k_ij`, the coefficient of `X_k` in `[X_i, X_j]`. Indices asserted in-range (callers hold `i,j,k < dimension()`). |

## Error model

| Condition | Error |
| :--- | :--- |
| `lie_bracket` / `lie_series` operands not both square of the same order | `MathError::domain_error` |
| `lie_series` `order < 0` | `MathError::domain_error` |
| `exponential_map` with non-square `A` or `terms < 1` | `MathError::domain_error` |
| Basis not linearly independent (singular Gram) in `structure_constants` / `adjoint_matrix` / `killing_form` | `MathError::domain_error` |
| Basis **not closed** under the bracket (a bracket leaves the span) | `MathError::domain_error` |
| `adjoint_matrix` / `killing_form` with `X` (or `Y`) not sharing the basis order | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

`StructureConstants::dimension()` is a total accessor and never errors;
`StructureConstants::at` asserts its indices in-range rather than returning an
error. `exponential_map`'s exactness is inherited from
[`matexp`](matexp.md) — a well-formed call still returns a `Result` that is only
the *exact* `e^A` under the nilpotency condition above.

## Worked examples

```cpp
import nimblecas.lie;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (std::int64_t v : row) rr.push_back(Rational::from_int(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};

// The standard so(3) generators L_x, L_y, L_z (real antisymmetric 3x3).
auto lx = mat({{0, 0, 0}, {0, 0, -1}, {0, 1, 0}});
auto ly = mat({{0, 0, 1}, {0, 0, 0}, {-1, 0, 0}});
auto lz = mat({{0, -1, 0}, {1, 0, 0}, {0, 0, 0}});
std::vector<Matrix> basis{lx, ly, lz};

// The bracket, exact over Q. [A,A] = 0 and the cyclic so(3) relations hold.
auto a = mat({{1, 2}, {3, 4}});
lie_bracket(a, a).value();                 // 2x2 zero matrix
lie_bracket(lx, ly).value();               // == L_z   ([L_x,L_y] = L_z)
lie_bracket(ly, lz).value();               // == L_x
lie_bracket(lz, lx).value();               // == L_y

// Non-square / mismatched-order operands fail.
lie_bracket(mat({{1, 2, 3}}), a).error();          // MathError::domain_error
lie_bracket(a, Matrix::identity(3)).error();       // MathError::domain_error

// Structure constants: c^k_ij = epsilon_{ijk} (Levi-Civita) for so(3).
auto sc = structure_constants(basis).value();
sc.dimension();                            // 3
sc.at(0, 1, 2);                            // 1    (c^z_xy)
sc.at(1, 0, 2);                            // -1   (c^z_yx, by antisymmetry)

// A non-closed basis is rejected: [L_x,L_y] = L_z leaves span{L_x,L_y}.
structure_constants(std::vector<Matrix>{lx, ly}).error();  // domain_error

// Adjoint representation: for so(3), ad_{L_x} in this basis equals L_x itself.
adjoint_matrix(lx, basis).value();         // == L_x
// ad_X is a linear operator on coordinates: ad_{L_x}(L_y coords) = L_z coords.
auto e1 = mat({{0}, {1}, {0}});            // coordinate vector of L_y
adjoint_matrix(lx, basis).value().multiply(e1).value();  // [[0],[0],[1]] = L_z

// Killing form: so(3) is compact simple, K(L_i,L_j) = -2 delta_ij.
killing_form(lx, lx, basis).value();       // -2
killing_form(lx, ly, basis).value();       // 0

// Lie series exp(t ad_L) f, truncated. L is nilpotent (L^2 = 0) here, so
// ad_L is nilpotent and the series terminates.
auto L = mat({{0, 1}, {0, 0}});
auto f = mat({{0, 0}, {1, 0}});
lie_series(L, f, ri(1), 0).value();        // == f                (order 0)
lie_series(L, f, ri(1), 2).value();        // [[1,-1],[1,-1]]     (all nonzero terms)
lie_series(L, f, ri(1), 5).value();        // [[1,-1],[1,-1]]     (stable past nilpotency)

// Adjoint action Ad_{exp L} f = exp(L) f exp(-L). For nilpotent L (L^2 = 0),
// exp(L) = I + L and exp(-L) = I - L exactly, so once `order` clears ad_L's
// nilpotency index the truncated series equals (I+L) f (I-L).
auto g       = mat({{1, 2}, {3, 4}});
auto exp_l   = Matrix::identity(2).add(L).value();          // I + L
auto exp_neg = Matrix::identity(2).subtract(L).value();     // I - L
auto conj    = exp_l.multiply(g).value().multiply(exp_neg).value();
adjoint_action_series(L, g, 6).value();    // == conj

// The exponential map, delegated to matexp. exp(L) for nilpotent L is exact.
exponential_map(L, 4).value();             // == I + L exactly (L^2 = 0)
exponential_map(a, 0).error();             // domain_error (terms >= 1 required)

// Lie series domain errors: mismatched order or negative order.
lie_series(a, Matrix::identity(3), ri(1), 2).error();  // domain_error
lie_series(a, a, ri(1), -1).error();                   // domain_error
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact dense linear algebra over `Q`
  that every algebra element, adjoint matrix, and bracket is built on.
- [`nimblecas.matexp`](matexp.md) — the matrix exponential `exponential_map`
  delegates to, and the source of its exact-iff-nilpotent honesty contract.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the entries,
  structure constants, and Killing form live in.
- [`nimblecas.complex`](complex.md) — a sibling exact-over-`Q` numeric module.
- [Documentation hub](../Index.md)
```