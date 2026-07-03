# `nimblecas.core` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/core/core.cppm`

The foundation module. It exports the error model (`MathError`, `Result<T>`,
`make_error`), the copy-on-write pointer (`CowPtr<T>`) that every immutable
`Expr` is built on, and the `cache_line_size` alignment constant. Everything
else in NimbleCAS depends, directly or transitively, on this module.

```cpp
import nimblecas.core;
```

## Constants

| Name | Type | Value | Purpose |
| :--- | :--- | :--- | :--- |
| `cache_line_size` | `std::size_t` | `64` | Alignment for hot data structures (Code Policy Rules 5/42). |

## Error model (railway-oriented programming)

Fallible operations never throw. They return a `Result<T>` that holds either a
value or a `MathError` (Rule 32). Callers thread the error through with the
`std::expected` monadic surface (`has_value()`, `value()`, `error()`,
`and_then`, …).

### `enum class MathError : std::uint8_t`

| Enumerator | `to_string_view` text |
| :--- | :--- |
| `division_by_zero` | `"division by zero"` |
| `undefined_value` | `"undefined value"` |
| `overflow` | `"overflow"` |
| `domain_error` | `"domain error"` |
| `syntax_error` | `"syntax error"` |
| `not_implemented` | `"not implemented"` |

```cpp
[[nodiscard]] constexpr auto to_string_view(MathError err) noexcept -> std::string_view;
```

Returns a human-readable message for an error. `constexpr` and `noexcept`;
returns `"unknown error"` for an out-of-range value. This is the text the
Python binding layer surfaces in the raised exception.

### `Result<T>` and `make_error`

```cpp
template <typename T>
using Result = std::expected<T, MathError>;

template <typename T>
[[nodiscard]] constexpr auto make_error(MathError err) -> Result<T>;
```

`Result<T>` is the canonical fallible return type across the codebase.
`make_error<T>(err)` is a terse factory for the error branch — it wraps the
error in `std::unexpected` so call sites read fluently:

```cpp
if (denominator == 0) {
    return make_error<Expr>(MathError::division_by_zero);
}
```

## `CowPtr<T>` — copy-on-write pointer

```cpp
template <typename T>
class CowPtr;
```

A value-semantic handle wrapping a `std::shared_ptr<T>`. The payload is treated
as **logically immutable**: only `read()` / `operator*` expose it, and they hand
back a `const` view, so shared instances are never mutated in place. Copying a
`CowPtr` is an O(1) atomic refcount bump. This is what makes copying an `Expr`
cheap and what lets independent subtrees be read concurrently.

### API

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| `make` | `static auto make(Args&&...) -> CowPtr` | Constructs a payload in place and returns a handle owning it. |
| `read` | `auto read() const noexcept -> const T&` | Const view of the payload. Asserts on an empty handle. |
| `write` | `auto write() -> T&` | Detaches a private copy first if the payload is shared (`use_count() > 1`), then returns a mutable reference to the now-unique copy. |
| `operator*` | `auto operator*() const noexcept -> const T&` | Alias for `read()`. |
| `operator->` | `auto operator->() const noexcept -> const T*` | Pointer to the payload (or `nullptr` if empty). |
| `operator bool` | `explicit operator bool() const noexcept` | `true` if the handle owns a payload. |
| `use_count` | `auto use_count() const noexcept -> long` | Number of handles sharing the payload. |
| `element_type` | type alias | `T`. |

### Why the payload is a non-`const` `T`

The `shared_ptr` deliberately owns a **non-`const`** `T`. Constructing the
object as `const T` and later `const_cast`-ing it to mutate (inside `write()`)
would be undefined behaviour (`[dcl.type.cv]`). Immutability is therefore
enforced by the API surface — the const-only read accessors — not by a `const`
dynamic type.

### Thread-safety

- Distinct `CowPtr` handles (e.g. one per thread, obtained by copy) may be read
  concurrently, and any one of them may call `write()` concurrently with reads
  of the **other** handles — the COW detach guarantees a writer never touches a
  payload another handle observes.
- A **single shared** `CowPtr` instance is **not** safe for concurrent
  `read()` + `write()`: `write()` reassigns the internal pointer, which races a
  concurrent reader of the same handle. Give each thread its own copy.

## Example

```cpp
import nimblecas.core;
using namespace nimblecas;

// Fallible construction threaded through Result.
Result<double> reciprocal = [](double x) -> Result<double> {
    if (x == 0.0) return make_error<double>(MathError::division_by_zero);
    return 1.0 / x;
}(4.0);

if (reciprocal) {
    // use *reciprocal
} else {
    std::println("error: {}", to_string_view(reciprocal.error()));
}

// Copy-on-write sharing.
auto a = CowPtr<std::vector<int>>::make(std::vector<int>{1, 2, 3});
auto b = a;                 // O(1) refcount bump; both observe {1, 2, 3}
b.write().push_back(4);     // detaches a private copy before mutating
// a still reads {1, 2, 3}; b reads {1, 2, 3, 4}
```

## See also

- [`nimblecas.symbolic`](symbolic.md) — builds immutable `Expr` on `CowPtr`.
- [Architecture overview](../architecture/overview.md) — the error model and data model in context.
- [Documentation hub](../Index.md)
