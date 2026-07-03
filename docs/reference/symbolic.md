# `nimblecas.symbolic` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/symbolic/symbolic.cppm`

The symbolic engine: immutable expression trees (`Expr`) and the foundational
Cohen primitives (`free_of`, `substitute`, structural equality and hashing).
Expressions are trees of `std::variant` nodes shared through `CowPtr` (no
inheritance hierarchy), so copies are cheap and safe to read across threads.

```cpp
import nimblecas.symbolic;
```

Depends on [`nimblecas.core`](core.md) and [`nimblecas.parallel`](parallel.md).

## `class Expr`

A copy-on-write handle to an immutable expression node. Every `Expr` memoizes,
at construction, its subtree node count (`size_`) and a structural hash
(`hash_`), so `size()` and `structural_hash()` are O(1) queries.

### Leaf factories

| Factory | Signature | Notes |
| :--- | :--- | :--- |
| `symbol` | `static auto symbol(std::string name) -> Expr` | A named variable. |
| `integer` | `static auto integer(std::int64_t value) -> Expr` | An exact integer constant. |
| `real` | `static auto real(double value) -> Expr` | An IEEE double constant. |
| `rational` | `static auto rational(std::int64_t num, std::int64_t den) -> Result<Expr>` | Exact fraction; **fallible** (see below). |

`rational` fails with `MathError::division_by_zero` on a zero denominator, and
with `MathError::overflow` if either argument is `INT64_MIN` (negating or
`gcd`-ing it would be UB). On success the result is **canonicalised**: the sign
is moved onto the numerator and the fraction is reduced by its gcd, so
structurally-equal fractions such as `2/4` and `1/2` compare equivalent.

### Compound factories

| Factory | Signature |
| :--- | :--- |
| `sum` | `static auto sum(std::vector<Expr> terms) -> Expr` |
| `product` | `static auto product(std::vector<Expr> factors) -> Expr` |
| `power` | `static auto power(Expr base, Expr exponent) -> Expr` |
| `apply` | `static auto apply(std::string name, std::vector<Expr> args) -> Expr` |

These construct raw (unsimplified) tree nodes. `apply` builds a function
application such as `sin(x)`. Run [`simplify`](simplify.md) to canonicalise.

### Fluent builders

| Method | Signature | Equivalent to |
| :--- | :--- | :--- |
| `add` | `auto add(const Expr& other) const -> Expr` | `Expr::sum({*this, other})` |
| `mul` | `auto mul(const Expr& other) const -> Expr` | `Expr::product({*this, other})` |
| `pow` | `auto pow(const Expr& exponent) const -> Expr` | `Expr::power(*this, exponent)` |

### Queries

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `node` | `auto node() const -> const ExprNode&` | The underlying variant node (for visiting). |
| `is_equivalent_to` | `auto is_equivalent_to(const Expr& other) const -> bool` | Structural (syntactic) equality — trees identical. |
| `to_string` | `auto to_string() const -> std::string` | Human-readable rendering. |
| `size` | `auto size() const noexcept -> std::size_t` | Memoized subtree node count (O(1)); drives the parallel cost gate. |
| `structural_hash` | `auto structural_hash() const noexcept -> std::size_t` | Memoized structural hash (O(1)); key for hash-consing / memoization. |

`operator==(const Expr&, const Expr&)` is defined as `is_equivalent_to`.

**Structural equality** is syntactic: two expressions are equivalent when their
trees are identical. It short-circuits in O(1) when the two handles share the
same COW node, and rejects in O(1) when the memoized hashes differ. Constants
are compared by kind and value; doubles are compared **bitwise** (via
`std::bit_cast`), so a `NaN` leaf equals itself and `+0.0`/`-0.0` stay distinct
— equality is syntactic, not IEEE. It is Cohen's automatic simplification (see
[`simplify`](simplify.md)) that maps *mathematically*-equal expressions to
identical trees.

`structural_hash()` is consistent with equality:
`a.is_equivalent_to(b)` implies `a.structural_hash() == b.structural_hash()`.

## Node kinds

An `ExprNode` wraps a `std::variant` of six alternatives:

```cpp
struct SymbolNode   { std::string name; };
struct ConstantNode { std::variant<std::int64_t, double,
                                    std::pair<std::int64_t, std::int64_t>> value; };
struct AddNode      { std::vector<Expr> terms; };
struct MulNode      { std::vector<Expr> factors; };
struct PowerNode    { Expr base; Expr exponent; };
struct FunctionNode { std::string name; std::vector<Expr> args; };

struct ExprNode {
    std::variant<SymbolNode, ConstantNode, AddNode, MulNode, PowerNode, FunctionNode> value;
};
```

`ConstantNode` unifies the three numeric kinds: integer (`int64`), real
(`double`), and exact rational (`pair<int64, int64>` = numerator, denominator).
Visitors over `ExprNode` are exhaustive by construction: a `static_assert` in
every `std::visit` chain fails to compile if a new alternative is added without
updating the visitor.

## Free functions

```cpp
[[nodiscard]] auto free_of(const Expr& u, const Expr& t) -> bool;
[[nodiscard]] auto substitute(const Expr& u, const Expr& t, const Expr& r) -> Expr;
[[nodiscard]] auto hash_value(const Expr& u) -> std::size_t;
```

- **`free_of(u, t)`** — `true` iff `u` does not contain the sub-expression `t`.
  Recurses structurally; leaves that are not `t` are trivially free of it.
- **`substitute(u, t, r)`** — returns a new tree with every occurrence of the
  sub-expression `t` replaced by `r`. Because the tree is immutable, the input
  `u` is unchanged. For sufficiently large subtrees (`size() >=
  parallel::parallel_cost_threshold`, i.e. 512) the independent children of a
  sum / product / function application are substituted **in parallel** via
  `parallel::transform_index_if`; the result is deterministic regardless of
  scheduling.
- **`hash_value(u)`** — O(1); returns the memoized `structural_hash()`. Used for
  the Python `__hash__` and for hash-consing / memoization keys.

## Immutability and parallelism

`Expr` is a persistent, immutable data structure. Every transformation builds a
*new* tree and never mutates an existing one, so independent subtrees carry zero
shared mutable state — the precondition for the safe, deterministic parallel
recursion used by `substitute`, [`simplify`](simplify.md), and
[`differentiate`](diff.md). See
[parallel tree computation](../architecture/parallel-tree-computation.md).

## Example

```cpp
import nimblecas.symbolic;
using namespace nimblecas;

const Expr x = Expr::symbol("x");
const Expr y = Expr::symbol("y");

// u = x^2 + x
const Expr u = x.pow(Expr::integer(2)).add(x);

bool has_x = !free_of(u, x);            // true
bool no_z  = free_of(u, Expr::symbol("z"));  // true

// substitute x -> y  =>  y^2 + y  (u is unchanged)
const Expr v = substitute(u, x, y);
assert(v.is_equivalent_to(y.pow(Expr::integer(2)).add(y)));

// exact rational is fallible
Result<Expr> half = Expr::rational(2, 4);   // canonicalises to 1/2
```

## See also

- [`nimblecas.simplify`](simplify.md) — canonicalises `Expr` trees.
- [`nimblecas.diff`](diff.md) — symbolic differentiation.
- [`nimblecas.parallel`](parallel.md) — the combinators `substitute` uses.
- [Documentation hub](../Index.md)
