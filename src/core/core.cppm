// NimbleCAS core module: fundamental types, alignment constants, copy-on-write
// pointer, and railway-oriented error handling.
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions).

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
// Wraps an immutable payload in std::shared_ptr<const T>. Copies are O(1) atomic
// refcount bumps and are safe to read concurrently across threads (the payload is
// const while shared). write() first detaches a private copy when the payload is
// shared, so mutation never races another reader.
// ---------------------------------------------------------------------------

template <typename T>
class CowPtr {
public:
    using element_type = T;

    CowPtr() = default;

    template <typename... Args>
    [[nodiscard]] static auto make(Args&&... args) -> CowPtr {
        CowPtr p;
        p.ptr_ = std::make_shared<const T>(std::forward<Args>(args)...);
        return p;
    }

    [[nodiscard]] auto read() const noexcept -> const T& { return *ptr_; }

    // Returns a mutable reference to a payload owned solely by this handle,
    // performing a copy-on-write detach first if the payload is currently shared.
    [[nodiscard]] auto write() -> T& {
        if (ptr_.use_count() > 1) {
            ptr_ = std::make_shared<const T>(*ptr_);
        }
        return const_cast<T&>(*ptr_);  // sole owner after detach: mutation is safe
    }

    [[nodiscard]] auto operator*() const noexcept -> const T& { return *ptr_; }
    [[nodiscard]] auto operator->() const noexcept -> const T* { return ptr_.get(); }
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }
    [[nodiscard]] auto use_count() const noexcept -> long { return ptr_.use_count(); }

private:
    std::shared_ptr<const T> ptr_{};
};

}  // namespace nimblecas
