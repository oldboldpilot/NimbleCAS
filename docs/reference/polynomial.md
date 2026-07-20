# `nimblecas.polynomial` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/polynomial/polynomial.cppm`

Dense univariate polynomial arithmetic over exact `int64` coefficients.
`coeffs[i]` is the coefficient of `x^i`, stored **trimmed** (no trailing zeros)
so the degree is unambiguous and equality is a plain vector compare. Ring
operations are overflow-checked and return `Result` (Rule 32).
`evaluate_batch` is the numeric fast path — it evaluates the polynomial at many
points at once through the [SIMD Horner kernel](simd.md).

```cpp
import nimblecas.polynomial;
```

Depends on [`core`](core.md) and [`simd`](simd.md).

## Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `Polynomial()` | The zero polynomial. |
| from coefficients | `explicit Polynomial(std::vector<std::int64_t> coefficients)` | `coefficients[i]` = coeff of `x^i`; trailing zeros trimmed. |
| `constant` | `static auto constant(std::int64_t c) -> Polynomial` | Degree-0 polynomial `c` (zero polynomial if `c == 0`). |
| `monomial` | `static auto monomial(std::int64_t coeff, std::size_t degree) -> Polynomial` | `coeff · x^degree`. |

## Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` for the zero polynomial. |
| `degree` | `auto degree() const noexcept -> std::int64_t` | Degree; **-1** for the zero polynomial (conventional). |
| `coefficient` | `auto coefficient(std::size_t i) const noexcept -> std::int64_t` | Coefficient of `x^i` (0 beyond the stored degree). |
| `leading_coefficient` | `auto leading_coefficient() const noexcept -> std::int64_t` | Highest-degree coefficient (0 for the zero polynomial). |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const std::int64_t>` | View of the trimmed coefficient vector. |
| `is_equal` | `auto is_equal(const Polynomial& other) const noexcept -> bool` | Exact coefficient-vector equality. |
| `to_string` | `auto to_string(std::string_view var = "x") const -> std::string` | Human-readable rendering. |

## Ring operations (overflow-checked, exact)

```cpp
[[nodiscard]] auto add(const Polynomial& other) const -> Result<Polynomial>;
[[nodiscard]] auto subtract(const Polynomial& other) const -> Result<Polynomial>;
[[nodiscard]] auto scale(std::int64_t s) const -> Result<Polynomial>;
[[nodiscard]] auto multiply(const Polynomial& other) const -> Result<Polynomial>;
```

Each returns `MathError::overflow` if any coefficient computation wraps `int64`.

## Number-theoretic / division operations

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `content` | `auto content() const -> Result<std::int64_t>` | gcd of the coefficients (`>= 0`; `0` for the zero polynomial). |
| `primitive_part` | `auto primitive_part() const -> Result<Polynomial>` | `this / content`, sign-normalised to a positive leading coefficient. |
| `pseudo_remainder` | `auto pseudo_remainder(const Polynomial& divisor) const -> Result<Polynomial>` | Pseudo-remainder `R` where `lc(divisor)^(deg-diff+1)·this = divisor·Q + R`, staying in `Z[x]` (no fractions). Fails on a zero divisor or `int64` overflow. |
| `gcd` | `auto gcd(const Polynomial& other) const -> Result<Polynomial>` | GCD in `Z[x]` via the primitive Euclidean PRS (content gcd × primitive gcd). |
| `derivative` | `auto derivative() const -> Result<Polynomial>` | Formal derivative `d/dx`. |
| `divide_exact` | `auto divide_exact(const Polynomial& divisor) const -> Result<Polynomial>` | Exact quotient `this / divisor` over `Z[x]`. |
| `square_free_factorization` | `auto square_free_factorization() const -> Result<std::vector<std::pair<Polynomial, std::int64_t>>>` | Yun's algorithm. |

- **`divide_exact`** returns `MathError::division_by_zero` on a zero divisor, and
  `MathError::domain_error` if the division is not exact over `Z` (a non-zero
  remainder, a lower-degree non-zero dividend, or a leading coefficient that does
  not divide evenly).
- **`square_free_factorization`** returns `(factor, multiplicity)` pairs such
  that the primitive part of `this` equals the product of `factor^multiplicity`,
  with each factor square-free and the factors pairwise coprime. Constants
  (`degree <= 0`) yield an empty list.

## Evaluation

```cpp
[[nodiscard]] auto evaluate(std::int64_t x) const -> Result<std::int64_t>;
[[nodiscard]] auto evaluate_batch(std::span<const float> xs) const -> std::vector<float>;
[[nodiscard]] auto evaluate_batch_into(std::span<const float> xs, std::span<float> out) const -> bool;
[[nodiscard]] auto evaluate_batch_parallel(std::span<const float> xs,
                                           std::size_t grain = default_batch_grain) const -> std::vector<float>;
[[nodiscard]] auto evaluate_batch_parallel_into(std::span<const float> xs, std::span<float> out,
                                                std::size_t grain = default_batch_grain) const -> bool;
```

- **`evaluate(x)`** — exact evaluation at an integer point via Horner's method,
  overflow-checked (`MathError::overflow` on wrap).
- **`evaluate_batch(xs)`** — evaluates the polynomial at many points at once in
  **floating point**, folding high-degree to low through the vectorised SIMD
  Horner kernel ([`simd::horner_step`](simd.md)). Coefficients are taken as
  `float` — this is the numeric path (NFR-1). Returns one `float` result per
  input point.
- **`evaluate_batch_into(xs, out)`** — the allocation-free fast path: writes
  `p(xs[i])` into `out[i]`, returning `false` (a no-op) when `out` is smaller
  than `xs`. Reusing one caller buffer across calls removes the per-call output
  allocation that `perf` flagged (page faults / sys time) — measured **~1.3×**
  over the allocating `evaluate_batch` on a large sweep. **Precondition:** `xs`
  and `out` must not overlap.

### Parallel batch evaluation

- **`evaluate_batch_parallel_into(xs, out, grain)`** / **`evaluate_batch_parallel(xs, grain)`**
  — shard `[0, xs.size())` across the fork-join runtime
  ([`nimblecas.parallel`](parallel.md); TBB on Linux/macOS, PPL on Windows) and
  run the identical serial SIMD Horner on each disjoint slice. The result is
  **bit-identical** to `evaluate_batch_into` (the shards partition the same
  indices; each runs the same per-element Horner — no reduction or reordering),
  a property asserted in `polynomial_tests`.

  These exist because a `perf` pass (Xeon Gold 6152, AVX-512) showed the serial
  `evaluate_batch_into` is **memory-latency bound** at one thread — IPC 0.31,
  ~56 % of cycles stalled on L3 misses, 80 % of samples in `horner_avx512` — so
  the AVX-512 ALU idles waiting on DRAM. Spreading shards across cores hides that
  latency: the repo's `cpu_bench` harness measures **~17–19×** over the serial
  in-place path on that 88-thread dual-socket box at 50M elements (default grain,
  ~46 GB/s aggregate — the memory-bandwidth ceiling). The factor is
  hardware-dependent; below `grain` (`default_batch_grain = 1<<16`) the call runs
  serially, so the serial path stays best for small `n`. **Precondition** (as
  `evaluate_batch_into`): `xs` and `out` must not overlap.

## Example

```cpp
import nimblecas.polynomial;
using namespace nimblecas;

// p = x^2 - 1
const Polynomial p{{-1, 0, 1}};
// q = x - 1
const Polynomial q{{-1, 1}};

auto g = p.gcd(q);                      // x - 1
auto exact = p.divide_exact(q);         // x + 1

auto y = p.evaluate(5);                 // 24 (exact)

std::array<float, 3> xs{0.0f, 1.0f, 2.0f};
auto ys = p.evaluate_batch(xs);         // {-1, 0, 3}
```

## See also

- [`nimblecas.polyexpr`](polyexpr.md) — bridges `Expr` and `Polynomial`.
- [`nimblecas.simd`](simd.md) — the Horner kernel behind `evaluate_batch`.
- [Documentation hub](../Index.md)
