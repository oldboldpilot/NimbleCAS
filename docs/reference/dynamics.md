# `nimblecas.dynamics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/dynamics/dynamics.cppm`

Exact stability analysis of **linear/affine autonomous systems** `dx/dt = A x (+ b)`
(ROADMAP §7.10). The whole module sits on top of the rational eigen substrate —
[`matrix`](matrix.md) and [`eigen`](eigen.md) over the exact `Rational` field —
and never touches floating point. Its distinguishing move is that stability is
decided from the **characteristic polynomial's coefficients** via the
Routh–Hurwitz criterion, not by finding roots: because the Routh array is built
in exact `Rational` arithmetic and only inspects signs, it settles
**irrational and complex spectra** that rational-root testing over `Q` cannot
see — for instance the pure-imaginary pair `±i` of a rotation. That is precisely
why Routh–Hurwitz is used here.

The honesty boundary. Everything stays inside exact `Rational` arithmetic, so
there is no numerical error and no root-finding tolerance anywhere. Two exact
capabilities are cleanly separated:

- `is_asymptotically_stable` is a **coefficient-level** decision. It is total
  over the whole spectrum (rational, irrational, or complex) because it never
  factors the polynomial — it only reads the Routh array's first column.
- `classify_equilibrium` gives a finer, human-readable verdict **only when the
  spectrum is fully rational** (the multiplicities returned by
  `rational_eigenvalues` sum to `n`). When part of the spectrum is
  irrational/complex — invisible over `Q` — it deliberately falls back to the
  Routh–Hurwitz verdict and never claims a `"center"`/`"focus"`, because
  rational eigenvalues cannot detect complex conjugate pairs.

Every fallible step is threaded on the [`Result`](core.md) railway. Overflow in
the exact arithmetic surfaces as `MathError::overflow` (an `int64`
numerator/denominator wrap inside a Routh entry), and every shape or
dimension violation surfaces as `MathError::domain_error`.

```cpp
import nimblecas.dynamics;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`eigen`](eigen.md).

## Free functions

All three entry points are free functions in namespace `nimblecas`; there is no
class to construct. Each takes exact `Rational` matrices and returns a
[`Result`](core.md).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fixed_point_affine` | `[[nodiscard]] auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix>` | The equilibrium (fixed point) of the affine map `x ↦ A x + b`, i.e. the exact solution of `(I − A) x = b` over `Q`. `a` must be square (`n × n`) and `b` an `n × 1` column, else `domain_error`. A non-isolated equilibrium — where `(I − A)` is singular — propagates as `domain_error` from the underlying solve. On success returns the `n × 1` column `x`. |
| `is_asymptotically_stable` | `[[nodiscard]] auto is_asymptotically_stable(const Matrix& a) -> Result<bool>` | Whether `dx/dt = A x` is asymptotically stable: `true` iff every eigenvalue of `A` has strictly negative real part (Hurwitz). Decided **exactly** from the characteristic polynomial via Routh–Hurwitz, with no root finding, so it also decides irrational/complex spectra. Requires a square matrix with `n ≥ 1`, else `domain_error`. |
| `classify_equilibrium` | `[[nodiscard]] auto classify_equilibrium(const Matrix& a) -> Result<std::string>` | A human-readable classification of the equilibrium at the origin of `dx/dt = A x`. Uses the exact rational sign pattern when the spectrum is fully rational; otherwise falls back to the Routh–Hurwitz verdict (see below). Requires a square matrix, else `domain_error`. |
| `is_perfect_square` | `[[nodiscard]] auto is_perfect_square(const Rational& r) -> bool` | Whether the rational `r` is the exact square of a rational. Exact over `Q`: an internal integer square root only seeds a search; every candidate is confirmed by exact overflow-guarded integer multiplication, so no floating point decides the verdict. A negative `r` is never a real square. |
| `classify_phase_portrait` | `[[nodiscard]] auto classify_phase_portrait(const Matrix& a) -> Result<PhasePortrait>` | The full **2×2** trace–determinant phase-portrait verdict (node / saddle / center / spiral·focus / star / degenerate node / non-isolated), decided exactly from the signs of `T`, `D`, `Δ` and from `is_perfect_square(Δ)`. Requires a `2×2` matrix, else `domain_error`. |
| `classify_linear_stability` | `[[nodiscard]] auto classify_linear_stability(const Matrix& a) -> Result<StabilityClassification>` | A coarser **nD** (`n ≥ 1`) stability class (sink / source / saddle / borderline) from the Routh–Hurwitz first-column sign-change count. Requires a square matrix with `n ≥ 1`, else `domain_error`. |
| `to_string` | `[[nodiscard]] auto to_string(PhaseType) -> std::string_view` (and `Stability`, `LinearStability` overloads) | Human-readable names for the classification enums. |

### `fixed_point_affine` — the affine equilibrium

Solves `(I − A) x = b` exactly over `Q`. The map's fixed point satisfies
`x = A x + b`, i.e. `(I − A) x = b`; the routine forms `I − A` and defers to the
matrix solve. Because everything is `Rational`, the returned column is the exact
equilibrium, not an approximation.

- Non-square `a` → `domain_error`.
- `b` whose row count differs from `a.rows()`, or `b` that is not a single
  column (`b.cols() != 1`) → `domain_error`.
- Singular `(I − A)` (a continuum of equilibria rather than an isolated one) →
  `domain_error`, propagated from the solve.

### `is_asymptotically_stable` — exact Routh–Hurwitz

The characteristic polynomial of `A` (monic, degree `n`) is taken from
[`eigen`](eigen.md), its coefficients laid out in descending order, and a Routh
array of `n + 1` rows is filled in exact `Rational` arithmetic. The system is
Hurwitz-stable **iff every entry of the array's first column is nonzero and
shares the leading term's (positive) sign**.

Two edge cases both mean at least one root lies **on or to the right of** the
imaginary axis, and so are reported as **not** asymptotically stable (`false`),
never as an error:

- a **zero pivot** in the first column (whether or not the rest of that row is
  nonzero); and
- a **fully-zero row** (imaginary-axis roots) — e.g. the second Routh row of a
  rotation `[[0,−1],[1,0]]`, whose spectrum `±i` rational-root testing over `Q`
  could not decide.

### `classify_equilibrium` — rational sign pattern, with an exact fallback

`rational_eigenvalues(A)` supplies the rational part of the spectrum with
multiplicities. If those multiplicities **sum to `n`** the spectrum is fully
rational and the verdict is exact from the sign pattern. A positive eigenvalue
forces instability regardless of the rest, so the positive cases are decided
before the zero (marginal) case (otherwise `diag(0, 1)`, which grows like
`e^t`, would be mislabelled marginal):

| Rational spectrum | Verdict string |
| :--- | :--- |
| a positive **and** a negative eigenvalue | `"saddle"` |
| all non-negative, some positive, no zero | `"unstable node"` |
| all non-negative, some positive, a zero | `"unstable (with marginal direction)"` |
| all non-positive, some negative, no zero | `"stable node"` |
| all non-positive, some negative, a zero | `"marginally stable"` |
| all eigenvalues zero (or `n == 0`) | `"degenerate/marginal"` |

Otherwise part of the spectrum is irrational/complex and cannot be seen over
`Q`, so the verdict defers to the exact `is_asymptotically_stable` result:

| Routh–Hurwitz verdict | Verdict string |
| :--- | :--- |
| asymptotically stable | `"asymptotically stable (spiral/node)"` |
| not asymptotically stable | `"unstable or marginal (non-rational spectrum)"` |

Because rational eigenvalues cannot detect complex conjugate pairs,
`"center"`/`"focus"` is never claimed from them.

### `classify_phase_portrait` — the exact 2×2 trace–determinant plane

For a planar linear system `ẋ = A x` with `A` a `2×2` rational matrix, the
equilibrium at the origin is classified **exactly** from three rationals:

- `T = tr(A)` — the trace,
- `D = det(A)` — the determinant,
- `Δ = T² − 4D` — the discriminant (the eigenvalues are `(T ± √Δ)/2`).

Everything below is decided by the **signs** of `T`, `D`, `Δ` and by whether `Δ`
is a **perfect rational square** (`is_perfect_square`). No root finding and no
floating point is involved.

| Region | `PhaseType` | Eigenvalues | `Stability` |
| :--- | :--- | :--- | :--- |
| `D < 0` | `saddle` | real, opposite sign (`Δ > 0` automatically) | `unstable` (always) |
| `D > 0`, `Δ > 0` | `node` | real, distinct, common sign of `T` | `stable` if `T < 0`, else `unstable` |
| `D > 0`, `Δ < 0`, `T ≠ 0` | `spiral` (focus) | complex pair, real part `T/2` | `stable` if `T < 0`, else `unstable` |
| `D > 0`, `Δ < 0`, `T = 0` | `center` | purely imaginary `±(√(−Δ)/2)·i` | `neutrally_stable` |
| `D > 0`, `Δ = 0`, `A = λI` | `star` (proper node) | repeated `T/2`, diagonalizable | `stable` if `T < 0`, else `unstable` |
| `D > 0`, `Δ = 0`, `A` defective | `degenerate_node` (improper) | repeated `T/2`, one eigenvector | `stable` if `T < 0`, else `unstable` |
| `D = 0` | `non_isolated` | a `0` eigenvalue; the other is `T` | `unstable` if `T > 0`, else `marginal` |

The `star` vs `degenerate_node` split (both are `D > 0, Δ = 0`) is decided
structurally: the equilibrium is a **star** exactly when `A` is the scalar
matrix `λI` — both off-diagonal entries zero and the two diagonal entries equal —
and an **improper/degenerate node** (a defective Jordan block) otherwise.

**The `PhasePortrait` struct** carries the exact `trace`, `determinant`,
`discriminant`, the `type`/`stability` verdict, boolean flags
(`complex_eigenvalues`, `repeated_eigenvalue`, `eigenvalues_rational`), and the
eigenvalues **only when they are rational**:

- Real case (`Δ ≥ 0`): `lambda1 = (T − √Δ)/2` and `lambda2 = (T + √Δ)/2` are
  filled iff `Δ` is a perfect square (equal when `Δ = 0`).
- Complex case (`Δ < 0`): the pair is `real_part ± imag_part·i`; `real_part =
  T/2` is **always** exact, while `imag_part = √(−Δ)/2` is filled iff `−Δ` is a
  perfect square.

**Honesty boundary.** When `Δ` (or `−Δ`) is not a perfect square the eigenvalues
are genuinely irrational (e.g. `(1 ± √5)/2`). Those optionals are then left
**empty** and `eigenvalues_rational` is `false` — the classification (`type`,
`stability`) is still fully exact, but no irrational eigenvalue is ever
approximated by a decimal. `description` is a ready-made one-liner such as
`"stable node"`, `"unstable spiral/focus"`, or `"center (neutrally stable)"`.

### `classify_linear_stability` — the coarse nD verdict

For `n ≥ 1` a full node/spiral/star taxonomy would need the actual eigenvalues,
which over `Q` are not available for irrational/complex spectra. Instead this
routine returns a **coarser** class computed from the characteristic
polynomial's Routh array: the number of roots with `Re > 0` equals the number of
**sign changes down the first column** (the Routh–Hurwitz theorem).

| `LinearStability` | Meaning | `rhp_count` |
| :--- | :--- | :--- |
| `sink` | every eigenvalue has `Re < 0` (asymptotically stable) | `0` |
| `source` | every eigenvalue has `Re > 0` | `n` |
| `saddle` | mixed: some `Re > 0` and some `Re < 0`, **none on the axis** | `1 … n−1` |
| `borderline` | the Routh array degenerates — a zero in the first column or a fully-zero row | `−1` (unknown) |

`sink`, `source` and `saddle` are the **regular** (hyperbolic) case where the
first column has no zeros; there the sign-change count is exact and there are no
imaginary-axis roots. `asymptotically_stable` mirrors `is_asymptotically_stable`
(`sink` iff `true`).

**Honesty boundary — the `borderline` case.** A zero pivot or a fully-zero row is
precisely the imaginary-axis / symmetric-root configuration the plain Routh array
cannot resolve (a center, a marginal line of equilibria, or a defective/unstable
case). Rather than guess, the verdict is `borderline` with `rhp_count = −1`:
deciding which of those it is needs the actual — possibly irrational or complex —
eigenvalues, which is out of scope over `Q`. This is the deliberate coarseness of
the nD classifier; use `classify_phase_portrait` for the exact `2×2` taxonomy.

## Error model

| Condition | Error |
| :--- | :--- |
| `fixed_point_affine`: `a` not square | `MathError::domain_error` |
| `fixed_point_affine`: `b.rows() != a.rows()` or `b.cols() != 1` | `MathError::domain_error` |
| `fixed_point_affine`: `(I − A)` singular (non-isolated equilibrium) | `MathError::domain_error` (from the solve) |
| `is_asymptotically_stable`: `a` not square, or `a.rows() < 1` | `MathError::domain_error` |
| `classify_equilibrium`: `a` not square | `MathError::domain_error` |
| `classify_phase_portrait`: `a` not `2×2` (non-square or `n ≠ 2`) | `MathError::domain_error` |
| `classify_linear_stability`: `a` not square, or `a.rows() < 1` | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps in a Routh entry, the characteristic polynomial, or the eigenvalue/solve substrate | `MathError::overflow` |

The two boundary cases of `is_asymptotically_stable` — a zero first-column entry
and a fully-zero row — are **results, not errors**: they return `false`. A
`domain_error` or `overflow` from the underlying `eigen`/`matrix` substrate is
propagated unchanged.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };

// Build an integer matrix from rows, and an n×1 integer column.
auto mat = [&](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (std::int64_t v : row) rr.push_back(ri(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};
auto col = [&](std::vector<std::int64_t> es) {
    std::vector<std::vector<std::int64_t>> rows;
    for (std::int64_t v : es) rows.push_back({v});
    return mat(std::move(rows));
};

// --- fixed_point_affine: exact equilibrium of x ↦ A x + b -----------------
// A = 0 ⇒ (I − A) = I ⇒ x = b.
auto x0 = fixed_point_affine(mat({{0, 0}, {0, 0}}), col({3, 5})).value();
x0.at(0, 0); x0.at(1, 0);                       // 3, 5

// A = 2I ⇒ (I − A) = −I ⇒ x = −b = [−1, −1].
auto x1 = fixed_point_affine(mat({{2, 0}, {0, 2}}), col({1, 1})).value();
x1.at(0, 0); x1.at(1, 0);                       // −1, −1

// A = I ⇒ (I − A) = 0, a non-isolated equilibrium ⇒ domain_error.
fixed_point_affine(mat({{1, 0}, {0, 1}}), col({1, 1})).error();  // domain_error

// --- is_asymptotically_stable: exact Routh–Hurwitz ------------------------
// Eigenvalues −1, −2 (λ² + 3λ + 2): Hurwitz.
is_asymptotically_stable(mat({{-1, 0}, {0, -2}})).value();       // true

// Eigenvalues 1, 1: not Hurwitz.
is_asymptotically_stable(mat({{1, 0}, {0, 1}})).value();         // false

// Rotation, eigenvalues ±i (λ² + 1): a fully-zero Routh row — the
// imaginary-axis case rational-root testing over Q could not decide.
is_asymptotically_stable(mat({{0, -1}, {1, 0}})).value();        // false

// Defective Jordan block, eigenvalue −1 (λ² + 2λ + 1): Hurwitz.
is_asymptotically_stable(mat({{-1, 1}, {0, -1}})).value();       // true

// --- classify_equilibrium: rational sign pattern + fallback ---------------
classify_equilibrium(mat({{-1, 0}, {0, -2}})).value();  // "stable node"
classify_equilibrium(mat({{1, 0}, {0, 2}})).value();    // "unstable node"
classify_equilibrium(mat({{-1, 0}, {0, 2}})).value();   // "saddle"

// A positive eigenvalue dominates a zero: diag(0, 1) grows like e^t.
classify_equilibrium(mat({{0, 0}, {0, 1}})).value();
                                    // "unstable (with marginal direction)"
// diag(0, −1): bounded but not asymptotically stable.
classify_equilibrium(mat({{0, 0}, {0, -1}})).value();   // "marginally stable"

// Non-rational spectrum (±i) ⇒ Routh–Hurwitz fallback string.
classify_equilibrium(mat({{0, -1}, {1, 0}})).value();
                                    // "unstable or marginal (non-rational spectrum)"

// Shape errors.
is_asymptotically_stable(mat({{1, 2, 3}, {4, 5, 6}})).error();  // domain_error
classify_equilibrium(mat({{1, 2, 3}, {4, 5, 6}})).error();      // domain_error

// --- classify_phase_portrait: the exact 2×2 trace–determinant plane -------
// Rotation [[0,−1],[1,0]]: T=0, D=1, Δ=−4 ⇒ purely imaginary ±i ⇒ CENTER.
auto center = classify_phase_portrait(mat({{0, -1}, {1, 0}})).value();
center.type == PhaseType::center;                 // true
center.stability == Stability::neutrally_stable;  // true
center.real_part.value(); center.imag_part.value();  // 0 and 1  (±i)

// [[−1,0],[0,−2]]: T=−3, D=2, Δ=1 ⇒ real −2, −1 ⇒ stable node.
auto sn = classify_phase_portrait(mat({{-1, 0}, {0, -2}})).value();
sn.type == PhaseType::node && sn.stability == Stability::stable;  // true
sn.lambda1.value(); sn.lambda2.value();            // −2 and −1
sn.description;                                     // "stable node"

// [[1,−1],[1,1]]: T=2, D=2, Δ=−4 ⇒ complex 1±i ⇒ unstable spiral/focus.
classify_phase_portrait(mat({{1, -1}, {1, 1}})).value().type;  // PhaseType::spiral

// [[2,0],[0,2]] = 2I: Δ=0 and scalar ⇒ unstable star (proper node).
classify_phase_portrait(mat({{2, 0}, {0, 2}})).value().type;   // PhaseType::star

// Irrational eigenvalues are NOT decimalised: [[0,1],[1,1]] has Δ=5.
auto irr = classify_phase_portrait(mat({{0, 1}, {1, 1}})).value();
irr.type == PhaseType::saddle;   // true; Δ=5 is not a perfect square
irr.eigenvalues_rational;        // false — lambda1/lambda2 left empty, no decimal

// --- classify_linear_stability: coarse nD Routh–Hurwitz verdict -----------
// diag(−1,−2,−3): λ³+6λ²+11λ+6, Hurwitz ⇒ sink.
auto sink = classify_linear_stability(mat({{-1,0,0},{0,-2,0},{0,0,-3}})).value();
sink.verdict == LinearStability::sink;  // true
sink.rhp_count;                          // 0

// diag(1,−2,−3): one Re>0, two Re<0 ⇒ saddle (rhp_count == 1).
classify_linear_stability(mat({{1,0,0},{0,-2,0},{0,0,-3}})).value().verdict;
                                         // LinearStability::saddle

// ±i and −1: the imaginary-axis pair degenerates the Routh array ⇒ borderline.
auto bd = classify_linear_stability(mat({{0,-1,0},{1,0,0},{0,0,-1}})).value();
bd.verdict == LinearStability::borderline;  // true
bd.rhp_count;                                // −1 (unknown at the boundary)
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrices, the solve
  behind `fixed_point_affine`, and the substrate for the spectrum.
- [`nimblecas.eigen`](eigen.md) — `characteristic_polynomial` and
  `rational_eigenvalues`, the exact spectral inputs this module consumes.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry,
  coefficient, and Routh-array value lives in.
- [`nimblecas.complex`](complex.md) — the exact Gaussian rationals `Q + Qi`,
  the sibling exact-numeric module for spectra that leave the reals.
- [Documentation hub](../Index.md)
