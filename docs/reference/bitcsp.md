# `nimblecas.bitcsp` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bitcsp/bitcsp.cppm`

A finite-domain constraint solver whose every hot path is **branchless and
word-parallel**. Each variable's live domain is a `Bitset` over its value range
`[0, k)`, and the two things that matter — AC-3 constraint propagation and the
N-queens search — run entirely on **bitwise operations with no value-by-value
branching**, so the CPU reference here has the exact control-flow shape a GPU
SIMT kernel would execute. It sits above `nimblecas.bitset` (the
domain container), [`parallel`](parallel.md) (the stateless subtree fan-out),
and [`core`](core.md) (the `Result` railway), and it is the branchless,
bit-parallel counterpart of the value-branching `nimblecas.csp` reference
solver — the two are cross-checked to agree.

The honesty boundary here is **combinatorial, not numerical**. Nothing in this
module is approximate: AC-3 computes an exact arc-consistency fixpoint and
`count_nqueens` returns an exact solution count. But both are **worst-case
exponential search problems**. AC-3 is only a *filter* — it prunes domains to a
fixpoint but does **not** decide satisfiability in general (a non-empty
arc-consistent CSP may still have no solution), and reaching a fixpoint is
polynomial in the domains but the underlying CSP is NP-complete. `count_nqueens`
enumerates a search tree whose size grows super-exponentially. The count is
returned as a `std::uint64_t`; the board size is capped at `n <= 32` so the
`n`-bit board mask and its diagonal shifts fit a 64-bit word, and that cap is a
hard `domain_error`, not a wrap.

```cpp
import nimblecas.bitcsp;
```

Depends on [`core`](core.md), `bitset`, and [`parallel`](parallel.md).

## The branchless model

**Support masks — the branchless propagator.** A binary constraint between
variables `x` and `y` is precompiled, once, from a `bool(int, int)` predicate
into two tables of `Bitset`s:

- `support_x_given_y[v]` = the set of `x`-values compatible with `y = v`, and
- `support_y_given_x[u]` = the set of `y`-values compatible with `x = u`.

Revising `D[x]` against `D[y]` is then pure bitwise: OR together
`support_x_given_y[v]` for every `v` currently in `D[y]` (a word-parallel
union), then AND that into `D[x]` (a word-parallel intersection). **No branch
inspects an individual `(x, y)` value pair at propagation time** — the per-pair
decisions were baked into the masks up front by `make_bit_constraint`. On a GPU
each union/intersection is a warp-coherent bitwise reduction, whereas a
value-branching AC-3 would diverge per pair.

**N-queens — the branchless-parallel showcase.** Rows are placed one at a time
with three bitmasks — occupied columns and the two diagonal frontiers — so the
free squares of a row are `~(cols | d1 | d2) & board_mask`; the lowest free
square is taken with `free & (-free)` (no conditional), and the recursion
carries shifted diagonal masks. Fanning the **first row's** placements across
`nimblecas::parallel::transform_index` makes each starting column an
independent, stateless subtree count — one thread per subtree — and the parallel
total is **identical** to the serial one for any worker count (summation is
associative and index order is preserved).

## `BitConstraint` — a binary constraint precompiled into support masks

A binary constraint between variables `x` and `y`. For a value `v` in `y`'s
range, `support_x_given_y[v]` is the `Bitset` of `x`-values compatible with
`y = v` (its capacity is `x`'s range size `kx`, and the table has `ky` entries);
`support_y_given_x` is the symmetric table for revising `y` against `x`. Build
one with `make_bit_constraint` — do **not** hand-populate the tables unless you
match the capacity/size invariants `bitcsp_validate` enforces.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `x` | `std::size_t` | Index of the first variable in the constraint's scope. |
| `y` | `std::size_t` | Index of the second variable. |
| `support_x_given_y` | `std::vector<Bitset>` | Indexed by a `y`-value `v`; each `Bitset` is over `x`'s range (capacity `kx`, `ky` entries). |
| `support_y_given_x` | `std::vector<Bitset>` | Indexed by an `x`-value `u`; each `Bitset` is over `y`'s range (capacity `ky`, `kx` entries). |

## `BitCsp` — a bitset-domain constraint problem

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `domains` | `std::vector<Bitset>` | `domains[i]` is variable `i`'s live `Bitset` over `[0, capacity_i)`. |
| `constraints` | `std::vector<BitConstraint>` | Binary constraints referencing those variables by index. |

## Free functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `make_bit_constraint` | `[[nodiscard]] auto make_bit_constraint(std::size_t x, std::size_t y, std::size_t kx, std::size_t ky, const std::function<bool(int, int)>& allowed) -> BitConstraint` | Compiles a binary constraint on `(x, y)` from `allowed(x_value, y_value)` into the two support-mask tables, given `x`'s range size `kx` and `y`'s range size `ky`. This is the **only** place the predicate is evaluated — `O(kx * ky)` once — after which propagation is pure bitwise. Infallible: returns a plain `BitConstraint`. |
| `ac3_bitset` | `[[nodiscard]] auto ac3_bitset(const BitCsp& csp) -> Result<std::optional<std::vector<Bitset>>>` | AC-3 arc consistency over the support masks. Returns the pruned per-variable `Bitset` domains, or an engaged `Result` holding `std::nullopt` when some domain is emptied (arc-inconsistent — hence unsatisfiable, but a **valid** result, not an error). Shape faults are `MathError::domain_error`. |
| `count_nqueens` | `[[nodiscard]] auto count_nqueens(int n, bool parallel) -> Result<std::uint64_t>` | Counts solutions to the `n`-queens problem with the branchless bitmask search. With `parallel` true the first row's `n` placements are fanned across `nimblecas::parallel::transform_index` as independent stateless subtree counts and summed; the result is **identical** to the serial count for any worker count. Returns `MathError::domain_error` unless `1 <= n <= 32`. |

### `ac3_bitset` semantics

The two-level return type separates three outcomes precisely:

- **Error** (`!r.has_value()`): the CSP is structurally malformed — a
  `MathError::domain_error` (see the error model below).
- **Inconsistent** (`r.has_value()` but `*r == std::nullopt`): AC-3 emptied some
  domain. The CSP is unsatisfiable, but this is a **valid** result, not an
  error, mirroring `nimblecas.csp`.
- **Consistent** (`r.has_value()` and `*r` engaged): `**r` is the vector of
  pruned domains. **Arc consistency is a necessary, not sufficient, condition
  for satisfiability** — a non-empty result still needs a search to confirm a
  solution exists.

The worklist is a FIFO seeded and re-queued in a fixed arc-index order, so the
reached fixpoint is **independent of scheduling** — deterministic for a given
input. Each constraint contributes two directed arcs (revise `x` given `y`, and
revise `y` given `x`); after a domain shrinks, every arc reading **into** that
domain is re-queued except the same-constraint reverse arc (a value dropped
under a constraint cannot have been the support for that same constraint's
reverse direction — a *different* constraint on the same ordered pair is **not**
skipped).

## Error model

| Condition | Error |
| :--- | :--- |
| `ac3_bitset` on an empty variable set (`domains` empty) | `MathError::domain_error` |
| `ac3_bitset` with a capacity-0 domain (a variable with no possible values) | `MathError::domain_error` |
| `ac3_bitset` with a constraint referencing an out-of-range variable (`x >= n` or `y >= n`) | `MathError::domain_error` |
| `ac3_bitset` with a support table whose size mismatches the domain it revises (`support_x_given_y.size() != ky` or `support_y_given_x.size() != kx`) | `MathError::domain_error` |
| `ac3_bitset` with a support `Bitset` whose capacity mismatches the revised domain | `MathError::domain_error` |
| `count_nqueens` with `n < 1` or `n > 32` | `MathError::domain_error` |

An **emptied domain during AC-3 is not an error** — it is a valid
`std::nullopt`. `make_bit_constraint` is infallible and returns a plain
`BitConstraint`; only `ac3_bitset` and `count_nqueens` return `Result`.

## Worked examples

```cpp
import nimblecas.bitcsp;
import nimblecas.bitset;
import nimblecas.core;
using namespace nimblecas;

// A full domain Bitset over [0, k): every value initially live.
auto full_domain = [](std::size_t k) -> Bitset {
    Bitset b(k);
    b.set_all();
    return b;
};

// --- AC-3: X, Y over {0,1,2} with X < Y ---------------------------------
// Arc consistency removes value 2 from D[X] (no larger Y) and value 0 from
// D[Y] (no smaller X), leaving D[X] = {0,1}, D[Y] = {1,2}.
BitCsp csp;
csp.domains = {full_domain(3), full_domain(3)};
csp.constraints.push_back(
    make_bit_constraint(0, 1, 3, 3, [](int a, int b) { return a < b; }));

auto r = ac3_bitset(csp);                 // Result<optional<vector<Bitset>>>
r.value().has_value();                    // true  (consistent)
const auto& d = *r.value();
d[0].set_bits();                          // {0, 1}
d[1].set_bits();                          // {1, 2}

// --- AC-3: an emptied domain is a valid nullopt, not an error -----------
// X, Y over the single value {0} with X < Y: value 0 in X has no support.
BitCsp bad;
bad.domains = {full_domain(1), full_domain(1)};
bad.constraints.push_back(
    make_bit_constraint(0, 1, 1, 1, [](int a, int b) { return a < b; }));
auto e = ac3_bitset(bad);
e.has_value();                            // true  (not an error)
e.value().has_value();                    // false (arc-inconsistent)

// --- AC-3 structural fault -> domain_error ------------------------------
BitCsp empty;
ac3_bitset(empty).error();                // MathError::domain_error (no variables)

// --- Branchless bitmask N-queens ----------------------------------------
count_nqueens(8, false).value();          // 92
count_nqueens(10, false).value();         // 724
count_nqueens(2, false).value();          // 0   (unsatisfiable, still a valid count)

// Parallel mode returns the identical count for any worker count.
count_nqueens(8, true).value();           // 92  (== serial)

// The board-size cap is a hard domain_error, not a wrap.
count_nqueens(33, false).error();         // MathError::domain_error
count_nqueens(0, false).error();          // MathError::domain_error
```

## See also

- [`nimblecas.parallel`](parallel.md) — the stateless `transform_index`
  fan-out that turns each first-row placement into an independent subtree
  count.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway both
  entry points return on.
- `nimblecas.csp` — the value-branching reference solver this module is
  cross-checked against (same 8-queens count of 92).
- [Documentation hub](../Index.md)
