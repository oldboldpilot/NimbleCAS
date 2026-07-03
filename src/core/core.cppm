// NimbleCAS core module: fundamental types, alignment constants, copy-on-write
// pointer, and railway-oriented error handling.
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions).

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.core;

import std;

export namespace nimblecas {

// Cache line size used for alignment of hot data structures (Code Policy Rules 5/42).
inline constexpr std::size_t cache_line_size = 64;

// ---------------------------------------------------------------------------
// Error model — railway-oriented programming via std::expected (Rule 32).
// ---------------------------------------------------------------------------

enum class MathError : std::uint8_t {
    division_by_zero,
    undefined_value,
    overflow,
    domain_error,
    syntax_error,
    not_implemented,
};

[[nodiscard]] constexpr auto to_string_view(MathError err) noexcept -> std::string_view {
    switch (err) {
        case MathError::division_by_zero: return "division by zero";
        case MathError::undefined_value:  return "undefined value";
        case MathError::overflow:         return "overflow";
        case MathError::domain_error:     return "domain error";
        case MathError::syntax_error:     return "syntax error";
        case MathError::not_implemented:  return "not implemented";
    }
    return "unknown error";
}

// Canonical result type: either a value of type T or a MathError.
template <typename T>
using Result = std::expected<T, MathError>;

// Convenience factory for the error branch, keeping call sites terse and fluent.
template <typename T>
[[nodiscard]] constexpr auto make_error(MathError err) -> Result<T> {
    return std::unexpected(err);
}

// ---------------------------------------------------------------------------
// Copy-on-write pointer (Rule 22).
//
// Wraps a payload in std::shared_ptr<T>. The payload is treated as logically
// immutable: only read()/operator* expose it, and they hand back a const view, so
// shared instances are never mutated in place. Copies are O(1) atomic refcount
// bumps. write() detaches a private copy first whenever the payload is shared
// (use_count > 1), then returns a mutable reference to that now-unique copy.
//
// The shared_ptr owns a NON-const T deliberately: creating the object as
// `const T` and later const_cast-ing it to mutate would be undefined behaviour
// ([dcl.type.cv]), so immutability is enforced by the API surface, not by a const
// dynamic type.
//
// Thread-safety: distinct CowPtr handles (e.g. one per thread, obtained by copy)
// may be read concurrently, and any one of them may call write() concurrently with
// reads of the OTHER handles — the COW detach guarantees writers never touch a
// payload another handle observes. A SINGLE shared CowPtr instance is NOT safe for
// concurrent read()+write(): write() reassigns the ptr_ member, which races a
// concurrent reader of the same handle. Give each thread its own copy.
// ---------------------------------------------------------------------------

template <typename T>
class CowPtr {
public:
    using element_type = T;

    CowPtr() = default;

    template <typename... Args>
    [[nodiscard]] static auto make(Args&&... args) -> CowPtr {
        CowPtr p;
        p.ptr_ = std::make_shared<T>(std::forward<Args>(args)...);
        return p;
    }

    [[nodiscard]] auto read() const noexcept -> const T& {
        assert(ptr_ && "CowPtr::read() on an empty handle");
        return *ptr_;
    }

    // Returns a mutable reference to a payload owned solely by this handle,
    // performing a copy-on-write detach first if the payload is currently shared.
    [[nodiscard]] auto write() -> T& {
        assert(ptr_ && "CowPtr::write() on an empty handle");
        if (ptr_.use_count() > 1) {
            ptr_ = std::make_shared<T>(*ptr_);  // detach a private copy before mutating
        }
        return *ptr_;
    }

    [[nodiscard]] auto operator*() const noexcept -> const T& { return read(); }
    [[nodiscard]] auto operator->() const noexcept -> const T* { return ptr_.get(); }
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }
    [[nodiscard]] auto use_count() const noexcept -> long { return ptr_.use_count(); }

private:
    std::shared_ptr<T> ptr_{};
};

// ---------------------------------------------------------------------------
// Variant alternative access (Rules 3/4/32).
// ---------------------------------------------------------------------------
// Pointer-free, exception-free counterpart to std::get_if. When the variant currently
// holds a T, returns the alternative wrapped in a std::unique_ptr<const T>; otherwise
// returns std::nullopt. The presence guard is std::optional — "maybe absent" is the
// honest meaning of a variant peek, so MathError stays reserved for real failures (a
// wrong alternative is not an error). Memory is RAII (Rule 4): unique_ptr is the
// default smart pointer — exclusive, caller-local ownership, no atomic refcount, no
// sharing. No raw pointer is ever formed: an aliasing pointer would require the
// address-of the alternative (a raw-pointer expression), so the alternative is copied
// into the unique_ptr instead. The engaged optional always holds a non-null pointer
// (make_unique never returns null), so a caller that took the `has_value()` branch may
// dereference **result without a null check. std::get cannot throw here — it is
// guarded by holds_alternative and thus unreachable on the wrong alternative (Rule 32).
template <typename T, typename... Ts>
[[nodiscard]] auto as(const std::variant<Ts...>& v)
    -> std::optional<std::unique_ptr<const T>> {
    if (std::holds_alternative<T>(v)) {
        return std::make_unique<const T>(std::get<T>(v));
    }
    return std::nullopt;
}

}  // namespace nimblecas
