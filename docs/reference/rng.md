# `nimblecas.rng` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/rng/rng.cppm`

A deterministic, reproducible, **counter-based** random-number substrate for the
Monte Carlo, MCMC and rejection-sampling layer (ROADMAP §7.8). It sits below the
sampling and stochastic-process consumers and above [`core`](core.md)'s error
model. The design is *stateless at its core*: the primitive
`counter_u64(key, counter)` maps an index to a well-distributed 64-bit value with
**no hidden state** via a Threefry-2×64 keyed bijection (20 rounds), while
`splitmix64` expands seeds and mixes the second key word. Because a draw depends
only on `(key, counter)` — never on thread count, scheduling, or how work was
partitioned — N workers can each own a disjoint slice of the counter space and
their concatenated outputs are **bit-identical** to a single worker walking the
whole range. There is no global mutable state and no time / entropy seeding:
identical seeds always reproduce identical results.

```cpp
import nimblecas.rng;
```

Depends on [`core`](core.md) (for `Result<T>` and `MathError`).

## The exactness boundary

This module is **numerical, not exact**. Unlike [`complex`](complex.md) or
[`ratpoly`](ratpoly.md), an `rng` draw is a raw 64-bit word (or an IEEE-754
`double` / `int64` derived from it), never a `Rational`. Three honesty
boundaries follow from that:

- **The 64-bit fold is not injective.** `counter_u64` runs the full 128-bit
  Threefry bijection but returns the fold `x0 ^ x1`, so distinct counters can
  collide at the birthday rate (~2³² draws). It is a Monte Carlo substrate — do
  **not** rely on it for collision-free unique IDs.
- **`uniform_int` carries modulo bias.** Mapping a 64-bit draw into `[lo, hi]`
  by `bits % width` has worst-case relative bias `range_size / 2⁶⁴` — negligible
  for the small ranges typical of Monte Carlo work, but growing for very wide
  ranges (a width near 2⁶³ can make some outcomes ~2× as likely). For unbiased
  draws over a huge range, reject-sample on top of `counter_u64`.
- **Non-finite widths are rejected, not silently NaN'd.** `uniform_double` guards
  the width `hi - lo`: an overflow to `±inf` (e.g. `lo = -1e308, hi = 1e308`)
  surfaces as `MathError::overflow` rather than leaking a `0 * inf = NaN` through
  `has_value()`.

## Free functions — the stateless core

The primitives are pure and stateless. `splitmix64`, `counter_u64` and
`uniform_unit` are **infallible** (`noexcept`, plain return); `uniform_double`
and `uniform_int` are **fallible** (`Result`) because they validate their
`[lo, hi]` bounds.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `splitmix64` | `[[nodiscard]] auto splitmix64(std::uint64_t x) noexcept -> std::uint64_t` | SplitMix64 finalizer: a pure, invertible bit-mixing function turning a sequentially incremented state into a well-distributed value. Used to expand a seed into a key and to mix the second key word inside the counter core. No error channel. |
| `counter_u64` | `[[nodiscard]] auto counter_u64(std::uint64_t key, std::uint64_t counter) noexcept -> std::uint64_t` | The parallelisable primitive. Deterministically maps `(key, counter)` to a well-distributed 64-bit value via a Threefry-2×64 keyed bijection (20 rounds), so adjacent counters decorrelate fully. Returns the fold `x0 ^ x1` (not injective — see boundary above). |
| `uniform_unit` | `[[nodiscard]] auto uniform_unit(std::uint64_t bits) noexcept -> double` | Maps a draw to a `double` in `[0, 1)` using the top 53 bits over 2⁵³, matching the 53-bit mantissa so every representable value in range is reachable and the result is never exactly `1.0`. |
| `uniform_double` | `[[nodiscard]] auto uniform_double(std::uint64_t bits, double lo, double hi) -> Result<double>` | Maps a draw to a `double` in `[lo, hi)`. `domain_error` if `hi < lo`; `overflow` if the width `hi - lo` is not finite. |
| `uniform_int` | `[[nodiscard]] auto uniform_int(std::uint64_t bits, std::int64_t lo, std::int64_t hi) -> Result<std::int64_t>` | Maps a draw to an integer in the **inclusive** range `[lo, hi]` via modulo reduction. `domain_error` if `hi < lo`. The full range `[INT64_MIN, INT64_MAX]` is handled specially (every draw is already in range). Carries the modulo bias described above. |

### The parallelism contract

`counter_u64` is stateless and referentially transparent. For a fixed `key`, the
sequence `counter_u64(key, 0), counter_u64(key, 1), …` is fully defined by the
index alone. Therefore **any** partition of the index range `0..N-1` into
disjoint sub-ranges — evaluated on independent threads and concatenated in index
order — yields exactly the same sequence as a single-threaded evaluation.
Parallel decompositions are reproducible regardless of thread count or
scheduling. This is the invariant the test suite exercises directly.

## `Rng` — sequential convenience generator

A thin, copyable, trivially cheap wrapper over the counter core holding
`{key, counter}`; every draw is `counter_u64(key, counter++)`. Construct one with
`Rng::seeded`, never directly (the constructor is private).

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `seeded` | `[[nodiscard]] static auto seeded(std::uint64_t seed) noexcept -> Rng` | Constructs a generator whose stream is `counter_u64(seeded-key, 0), (…,1), …`. The key is derived from `seed` via `splitmix64`, so nearby seeds still produce well-separated streams. Equal seeds ⇒ equal streams. |
| `next_u64` | `[[nodiscard]] auto next_u64() noexcept -> std::uint64_t` | Next raw 64-bit draw; advances the internal counter. |
| `split` | `[[nodiscard]] auto split(std::uint64_t index) const noexcept -> Rng` | Returns an independent child stream whose key is derived from `(this key, index)` through the counter core. Distinct indices yield independent streams, and children are independent of the parent — JAX-style tree splitting for parallel decomposition. Does **not** advance this generator. |
| `next_unit` | `[[nodiscard]] auto next_unit() noexcept -> double` | Consumes one draw; returns a `double` in `[0, 1)` (`uniform_unit`). |
| `next_double` | `[[nodiscard]] auto next_double(double lo, double hi) -> Result<double>` | Consumes one draw; returns a `double` in `[lo, hi)` (`uniform_double`). |
| `next_int` | `[[nodiscard]] auto next_int(std::int64_t lo, std::int64_t hi) -> Result<std::int64_t>` | Consumes one draw; returns an integer in the inclusive `[lo, hi]` (`uniform_int`). |
| `key` | `[[nodiscard]] auto key() const noexcept -> std::uint64_t` | Observer: the generator's key (for reproducibility checks / checkpointing). |
| `counter` | `[[nodiscard]] auto counter() const noexcept -> std::uint64_t` | Observer: the current counter position. |

## Error model

| Condition | Error |
| :--- | :--- |
| `uniform_double` / `next_double` with `hi < lo` | `MathError::domain_error` |
| `uniform_int` / `next_int` with `hi < lo` | `MathError::domain_error` |
| `uniform_double` / `next_double` with a non-finite width `hi - lo` | `MathError::overflow` |

`splitmix64`, `counter_u64`, `uniform_unit`, `next_u64`, `next_unit`, `seeded`,
`split`, `key` and `counter` are total and `noexcept` — they have no error
channel.

## Worked examples

```cpp
import nimblecas.rng;
import nimblecas.core;
using namespace nimblecas;

// Determinism: equal seeds reproduce the stream exactly.
auto a = Rng::seeded(42);
auto b = Rng::seeded(42);
a.next_u64() == b.next_u64();             // true, and stays true for every draw

// The parallelism invariant: a partitioned run matches the sequential run.
constexpr std::uint64_t key = 0xABCDEF0123456789ULL;
std::vector<std::uint64_t> sequential;
for (std::uint64_t i = 0; i < 1000; ++i)
    sequential.push_back(counter_u64(key, i));
// Any disjoint cover of [0, 1000), concatenated in index order, is element-wise
// identical to `sequential` — independent of how the range was sliced.

// JAX-style splitting: independent children that do not disturb the parent.
auto parent = Rng::seeded(7);
auto child0 = parent.split(0);
auto child1 = parent.split(1);
parent.counter() == 0;                    // true — split does not advance parent
child0.next_u64() != child1.next_u64();   // independent streams
Rng::seeded(7).split(0).next_u64();       // reproducible: same as child0's draw

// Distributions over [0, 1).
auto rng = Rng::seeded(123);
double u = rng.next_unit();               // 0.0 <= u < 1.0
uniform_unit(0) == 0.0;                                       // true
uniform_unit(std::numeric_limits<std::uint64_t>::max()) < 1.0;  // strictly below 1

// Inclusive integer range, both endpoints reachable.
auto r = rng.next_int(-5, 9).value();     // in [-5, 9]
uniform_int(0xDEADBEEF, 3, 3).value();    // 3 (degenerate single-point range)

// Error railway.
uniform_double(0, 5.0, 1.0).error();      // MathError::domain_error   (hi < lo)
uniform_int(0, 10, 2).error();            // MathError::domain_error   (hi < lo)
uniform_double(0, -1e308, 1e308).error(); // MathError::overflow (non-finite width)
uniform_double(0x1234567890ABCDEFULL, -2.0, 2.0).value();  // in [-2.0, 2.0)

// The mixing primitive: deterministic, separates adjacent inputs.
splitmix64(0) == splitmix64(0);           // true
splitmix64(0) != splitmix64(1);           // true
```

## See also

- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway the
  fallible distributions return on.
- [`nimblecas.stats`](stats.md) — the distribution / sampling layer that consumes
  these draws.
- [`nimblecas.parallel`](parallel.md) — the partitioning substrate that the
  stateless counter core makes deterministic.
- [Documentation hub](../Index.md)
