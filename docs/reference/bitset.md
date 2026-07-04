# `nimblecas.bitset` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bitset/bitset.cppm`

A fixed-capacity set of bits `[0, capacity)` packed into an array of 64-bit
words. THE POINT is **branchless bit-parallelism**: every hot operation is a
straight, unconditional loop over the whole word array applying **one** bitwise
op per word, so 64 set elements are combined per machine word with no per-bit,
data-dependent branching. That is exactly the shape a GPU wants — a warp of
threads each owning a word (or a lane) runs the same instruction with no
divergence — so this CPU reference maps directly onto SIMT. It is the CPU
substrate beneath the branchless CSP / GPU-style regular workloads: a
value-semantic domain representation whose set operations never branch on data.

The honesty boundary here is not exact-vs-numerical (there is no arithmetic and
no floating point) but **representation and precondition**: capacity is fixed at
construction (no dynamic growth); indices are `[0, capacity)` and out-of-range
access is a **debug-asserted precondition**, not a checked error; and the binary
combinators split into a `Result`-returning form that rejects a capacity
mismatch and a `noexcept` in-place form that **assumes** equal capacity. The one
structural invariant — bits at indices `>= capacity` are always `0` — is
preserved by every operation, so the reductions never observe a phantom tail
bit.

```cpp
import nimblecas.bitset;
```

Depends on [`core`](core.md) (for `Result<T>`, `make_error`, `MathError`).

## The invariant and its consequences

**Invariant:** bits at indices `>= capacity` are always `0`. The only operation
that could introduce a high bit is `set_all` (it writes `~0` into every word); it
immediately masks the final word back down with a precomputed `last_mask_`. Every
combinator preserves the invariant, because `AND`/`OR`/`XOR`/`ANDNOT` of two
operands whose out-of-range bits are `0` leaves those positions `0`.

The payoff is that `count()`, `all()`, `first_set()`, and `set_bits()` need **no
special-casing of the tail word** — they scan every word uniformly and still
never see a bit past `capacity`. Word layout is fixed little-endian: word `w`
holds bits `64w .. 64w+63`, and bit `b` of a word has value `1 << b`.

**Determinism:** the ops are pure functions of the bit contents; `set_bits()`
and `first_set()` scan in ascending index order, so results never depend on
scheduling or word layout beyond that fixed convention.

## Error model at a glance (railway, Rule 32)

The pure bitwise ops **cannot fail** and are plain / `noexcept`. Only the
`Result`-returning combinators are fallible, and only on a capacity mismatch.
Nothing in this module throws. See [Error model](#error-model) below.

## `Bitset` — a fixed-capacity, word-packed set of bits

Copyable and movable. Two bitsets compare equal iff they have the **same
capacity** and the **same bit contents**.

### Construction

| Constructor | Signature | Behavior |
| :--- | :--- | :--- |
| default | `Bitset()` | A capacity-`0` bitset (no words). Present so `Bitset` is a well-behaved value type in containers (`std::vector<Bitset>`, `std::optional<Bitset>`); has no set bits. |
| capacity | `explicit Bitset(std::size_t capacity)` | A bitset able to hold indices `[0, capacity)`, all bits initially clear. Allocates `ceil(capacity / 64)` words. Only the vector allocation can fail. |

### Capacity queries

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `capacity` | `auto capacity() const noexcept -> std::size_t` | The number of representable indices. |
| `word_count` | `auto word_count() const noexcept -> std::size_t` | Number of 64-bit words allocated, `ceil(capacity / 64)`. |

### Single-bit access

Branchless by construction. `test()` shifts the owning word down and masks the
low bit; `set()`/`reset()` OR-in / AND-out a one-hot mask. Each takes an index
`i` with the **precondition** `i < capacity` (debug-asserted; not a checked
error).

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `test` | `[[nodiscard]] auto test(std::size_t i) const noexcept -> bool` | True iff bit `i` is set: `(word >> b) & 1`. Precondition `i < capacity`. |
| `set` | `auto set(std::size_t i) noexcept -> void` | Sets bit `i` (idempotent). Precondition `i < capacity`. |
| `reset` | `auto reset(std::size_t i) noexcept -> void` | Clears bit `i` (idempotent). Precondition `i < capacity`. |
| `set_all` | `auto set_all() noexcept -> void` | Sets every bit in `[0, capacity)`. Writes `~0` word-parallel, then masks the unused high bits of the final word to preserve the invariant. |
| `clear` | `auto clear() noexcept -> void` | Clears every bit. Straight word-parallel zero-fill. |

### Word-parallel combinators

Each in-place form is a single unconditional loop over the words — the
branchless / SIMT-friendly core. It is `noexcept` and **assumes equal capacity**
(debug-asserted). The `Result` forms wrap the same word-parallel apply with a
capacity check, return a **new** bitset, and leave both operands untouched.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `and_inplace` | `auto and_inplace(const Bitset& other) noexcept -> void` | `this &= other` (intersection). Precondition: equal capacity. |
| `or_inplace` | `auto or_inplace(const Bitset& other) noexcept -> void` | `this |= other` (union). Precondition: equal capacity. |
| `xor_inplace` | `auto xor_inplace(const Bitset& other) noexcept -> void` | `this ^= other` (symmetric difference). Precondition: equal capacity. |
| `andnot_inplace` | `auto andnot_inplace(const Bitset& other) noexcept -> void` | `this &= ~other` (set difference: keep bits in `this` not in `other`). Precondition: equal capacity. Preserves the invariant with no tail masking. |
| `and_with` | `[[nodiscard]] auto and_with(const Bitset& other) const -> Result<Bitset>` | New bitset `this & other`; `domain_error` on capacity mismatch. |
| `or_with` | `[[nodiscard]] auto or_with(const Bitset& other) const -> Result<Bitset>` | New bitset `this | other`; `domain_error` on capacity mismatch. |
| `xor_with` | `[[nodiscard]] auto xor_with(const Bitset& other) const -> Result<Bitset>` | New bitset `this ^ other`; `domain_error` on capacity mismatch. |
| `andnot_with` | `[[nodiscard]] auto andnot_with(const Bitset& other) const -> Result<Bitset>` | New bitset `this & ~other`; `domain_error` on capacity mismatch. |

### Reductions and queries

All word-parallel and branch-light; all correct with no tail special-case
because high bits are invariantly `0`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `count` | `[[nodiscard]] auto count() const noexcept -> std::size_t` | Number of set bits: `std::popcount` summed over the words (one hardware `POPCNT` per word). |
| `any` | `[[nodiscard]] auto any() const noexcept -> bool` | True iff any bit is set. OR-reduces the words and tests the accumulator once. |
| `none` | `[[nodiscard]] auto none() const noexcept -> bool` | `!any()`. |
| `all` | `[[nodiscard]] auto all() const noexcept -> bool` | True iff every index in `[0, capacity)` is set (vacuously true for capacity `0`). Computed as `count() == capacity`. |
| `first_set` | `[[nodiscard]] auto first_set() const noexcept -> std::size_t` | Index of the lowest set bit, or **`capacity` when none is set**. Scans words ascending and uses `std::countr_zero` (one hardware `CTZ`) inside the first non-zero word. |
| `set_bits` | `[[nodiscard]] auto set_bits() const -> std::vector<std::size_t>` | The indices of all set bits, **ascending**. Uses the branchless lowest-bit idiom (`w & (~w + 1)` isolates the lowest set bit, `w &= w - 1` clears it): one `CTZ` per set bit, nothing per clear bit. |

### Equality

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `operator==` | `[[nodiscard]] auto operator==(const Bitset& other) const noexcept -> bool` | Value equality: **same capacity and identical bit contents**. Bitsets with the same bits but different capacity compare unequal. |

## Error model

The only checked failure in the module is a capacity mismatch on a
`Result`-returning combinator.

| Condition | Error |
| :--- | :--- |
| `and_with` / `or_with` / `xor_with` / `andnot_with` on operands of **different capacity** | `MathError::domain_error` |

Everything else is either infallible or a **debug-asserted precondition**, not a
runtime error:

- `test` / `set` / `reset` with `i >= capacity` violate a precondition (asserted
  when `!NDEBUG`); they do **not** return a `Result` and must not be called
  out of range.
- The `*_inplace` combinators **assume** equal capacity (asserted); mismatched
  capacity is a precondition violation, not a `domain_error`. Use the `_with`
  forms when the capacities are not statically known to match.
- `count`, `any`, `none`, `all`, `first_set`, `set_bits`, `set_all`, `clear`,
  and the capacity queries are total and `noexcept` (except `set_bits`, which
  allocates its result vector).

## Worked examples

```cpp
import nimblecas.bitset;
import nimblecas.core;
using nimblecas::Bitset;
using nimblecas::MathError;
using Indices = std::vector<std::size_t>;

// Single-bit access; set/reset are idempotent.
Bitset b(10);
b.set(0);
b.set(3);
b.set(9);
b.test(0);                       // true
b.test(1);                       // false
b.count();                       // 3
b.reset(3);
b.count();                       // 2

// Bulk fill and clear.
Bitset f(5);
f.set_all();
f.all();                         // true
f.count();                       // 5
f.clear();
f.none();                        // true

// first_set: lowest set index, or capacity when empty.
Bitset empty(8);
empty.first_set();               // 8  (== capacity, i.e. "none")
Bitset s(8);
s.set(5); s.set(6);
s.first_set();                   // 5

// set_bits enumerates ascending, across word boundaries.
Bitset w(130);
for (std::size_t i : {0u, 63u, 64u, 65u, 129u}) w.set(i);
w.word_count();                  // 3   (130 bits -> three 64-bit words)
w.count();                       // 5
w.set_bits();                    // {0, 63, 64, 65, 129}

// Word-parallel combinators. a = {0,1,2,3}, b = {2,3,4,5} over capacity 8.
Bitset a(8); for (std::size_t i : {0u,1u,2u,3u}) a.set(i);
Bitset c(8); for (std::size_t i : {2u,3u,4u,5u}) c.set(i);

a.and_with(c).value().set_bits();     // {2, 3}
a.or_with(c).value().set_bits();      // {0, 1, 2, 3, 4, 5}
a.xor_with(c).value().set_bits();     // {0, 1, 4, 5}
a.andnot_with(c).value().set_bits();  // {0, 1}   (a & ~c)
a.set_bits();                         // {0, 1, 2, 3}  (Result forms leave a untouched)

// In-place form agrees with the Result form (equal capacity assumed).
Bitset x = a;
x.and_inplace(c);
x == a.and_with(c).value();           // true

// Capacity mismatch on a Result combinator surfaces as domain_error.
Bitset small(8);
Bitset big(16);
small.and_with(big).error();          // MathError::domain_error

// Equality is capacity-and-contents.
Bitset p(8);  p.set(1); p.set(3);
Bitset q(16); q.set(1); q.set(3);     // same bits, different capacity
p == q;                               // false

// Capacity edges: 0 is vacuously all(), never any().
Bitset zero(0);
zero.all();                           // true
zero.any();                           // false
zero.first_set();                     // 0  (== capacity)

// set_all masks the unused tail: capacity 65 yields count 65, not 128.
Bitset full65(65);
full65.set_all();
full65.word_count();                  // 2
full65.count();                       // 65
```

## See also

- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway the
  combinators return through.
- [`nimblecas.simd`](simd.md) — the sibling CPU-substrate module: branchless
  word-parallelism here, elementwise SIMD kernels there.
- [`nimblecas.gpu`](gpu.md) — the SIMT target this branchless, divergence-free
  design maps onto.
- [Documentation hub](../Index.md)
