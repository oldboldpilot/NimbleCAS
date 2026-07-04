// NimbleCAS fixed-capacity bit set substrate (branchless / bit-parallel).
// @author Olumuyiwa Oluwasanmi
//
// A fixed-capacity set of bits stored as an array of 64-bit words. THE POINT is
// BRANCHLESS BIT-PARALLELISM: every hot operation is a straight, unconditional loop
// over the whole word array applying ONE bitwise op per word — 64 set elements are
// combined per machine word with no per-bit, data-dependent branching. That is exactly
// the shape a GPU wants: a warp of threads each owning a word (or a lane) runs the SAME
// instruction with no divergence, so this CPU reference maps directly onto SIMT.
//
// INVARIANT: bits at indices >= capacity are always 0. The only way high bits could
// appear is set_all (which writes ~0 into every word); it immediately masks the last
// word back down. Every combinator preserves the invariant because AND/OR/XOR/ANDNOT of
// two operands whose out-of-range bits are 0 leaves those positions 0 — so count(),
// all(), first_set() and set_bits() never observe a phantom bit and need no special-casing
// of the tail word in their hot loops.
//
// DETERMINISM: the ops are pure functions of the bit contents; set_bits() and first_set()
// scan in ascending index order, so results never depend on scheduling or word layout
// beyond the fixed little-endian (word 0 holds bits 0..63) convention used throughout.
//
// ERRORS (railway, Rule 32): the pure bitwise ops cannot fail and are plain/noexcept. The
// binary combinators come in two forms — a Result-returning form that rejects a
// capacity mismatch with domain_error, and a noexcept in-place form that ASSUMES equal
// capacity (asserted in debug). Nothing here throws.

module;
#include <cassert>  // assert (unavailable via `import std`); active when !NDEBUG

export module nimblecas.bitset;

import std;
import nimblecas.core;

export namespace nimblecas {

// A fixed-capacity set of bits [0, capacity) packed into 64-bit words (word w holds bits
// 64w .. 64w+63, bit b of a word is value 1<<b). Copyable and movable; two bitsets compare
// equal iff they have the same capacity and the same bit contents.
class Bitset {
public:
    // A capacity-0 bitset (no words). Present so Bitset is a well-behaved value type in
    // containers (std::vector<Bitset>, std::optional<Bitset>); has no set bits.
    Bitset() = default;

    // A bitset able to hold indices [0, capacity), all bits initially clear. Allocates
    // ceil(capacity / 64) words. `noexcept`-adjacent: only the vector allocation can fail.
    explicit Bitset(std::size_t capacity)
        : capacity_(capacity),
          words_((capacity + 63) / 64, std::uint64_t{0}),
          last_mask_(compute_last_mask(capacity)) {}

    [[nodiscard]] auto capacity() const noexcept -> std::size_t { return capacity_; }
    [[nodiscard]] auto word_count() const noexcept -> std::size_t { return words_.size(); }

    // --- Single-bit access. test() is branchless by construction: it shifts the owning
    // word down and masks the low bit, with no conditional on the bit's value. set()/reset()
    // OR-in / AND-out a one-hot mask — also branchless given a valid index. -----------------

    // True iff bit i is set. Branchless: (word >> b) & 1. Precondition: i < capacity.
    [[nodiscard]] auto test(std::size_t i) const noexcept -> bool {
        assert(i < capacity_ && "Bitset::test index out of range");
        return ((words_[i >> 6] >> (i & 63)) & std::uint64_t{1}) != 0;
    }

    // Sets bit i (idempotent). Precondition: i < capacity.
    auto set(std::size_t i) noexcept -> void {
        assert(i < capacity_ && "Bitset::set index out of range");
        words_[i >> 6] |= (std::uint64_t{1} << (i & 63));
    }

    // Clears bit i (idempotent). Precondition: i < capacity.
    auto reset(std::size_t i) noexcept -> void {
        assert(i < capacity_ && "Bitset::reset index out of range");
        words_[i >> 6] &= ~(std::uint64_t{1} << (i & 63));
    }

    // Sets every bit in [0, capacity). Writes ~0 into every word word-parallel, then masks
    // the unused high bits of the final word to keep the "no bits >= capacity" invariant.
    auto set_all() noexcept -> void {
        for (std::uint64_t& w : words_) {
            w = ~std::uint64_t{0};
        }
        if (!words_.empty()) {
            words_.back() &= last_mask_;
        }
    }

    // Clears every bit. Straight word-parallel zero-fill.
    auto clear() noexcept -> void {
        for (std::uint64_t& w : words_) {
            w = std::uint64_t{0};
        }
    }

    // --- Word-parallel combinators. Each in-place form is a single unconditional loop over
    // the words (the branchless / SIMT-friendly core); the Result forms wrap them with a
    // capacity check. In-place forms ASSUME equal capacity (asserted). -----------------------

    // this &= other. Precondition: equal capacity.
    auto and_inplace(const Bitset& other) noexcept -> void {
        assert(capacity_ == other.capacity_ && "Bitset::and_inplace capacity mismatch");
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] &= other.words_[i];
        }
    }

    // this |= other. Precondition: equal capacity.
    auto or_inplace(const Bitset& other) noexcept -> void {
        assert(capacity_ == other.capacity_ && "Bitset::or_inplace capacity mismatch");
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] |= other.words_[i];
        }
    }

    // this ^= other. Precondition: equal capacity.
    auto xor_inplace(const Bitset& other) noexcept -> void {
        assert(capacity_ == other.capacity_ && "Bitset::xor_inplace capacity mismatch");
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] ^= other.words_[i];
        }
    }

    // this &= ~other (set difference: keep bits in this that are NOT in other). Precondition:
    // equal capacity. Because other's out-of-range bits are 0, ~other has 1s there, but this
    // is already 0 there — so the invariant is preserved with no tail masking.
    auto andnot_inplace(const Bitset& other) noexcept -> void {
        assert(capacity_ == other.capacity_ && "Bitset::andnot_inplace capacity mismatch");
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] &= ~other.words_[i];
        }
    }

    // Result-returning combinators: same-capacity is required, else domain_error. Each
    // returns a NEW bitset and leaves both operands untouched.
    [[nodiscard]] auto and_with(const Bitset& other) const -> Result<Bitset> {
        return combine(other, [](std::uint64_t a, std::uint64_t b) { return a & b; });
    }
    [[nodiscard]] auto or_with(const Bitset& other) const -> Result<Bitset> {
        return combine(other, [](std::uint64_t a, std::uint64_t b) { return a | b; });
    }
    [[nodiscard]] auto xor_with(const Bitset& other) const -> Result<Bitset> {
        return combine(other, [](std::uint64_t a, std::uint64_t b) { return a ^ b; });
    }
    [[nodiscard]] auto andnot_with(const Bitset& other) const -> Result<Bitset> {
        return combine(other, [](std::uint64_t a, std::uint64_t b) { return a & ~b; });
    }

    // --- Reductions / queries. All word-parallel and branch-light. ---------------------------

    // Number of set bits: std::popcount summed over the words (a single hardware POPCNT per
    // word on CPUs, a warp reduction on a GPU). Correct with no tail special-case because
    // high bits are invariantly 0.
    [[nodiscard]] auto count() const noexcept -> std::size_t {
        std::size_t total = 0;
        for (const std::uint64_t w : words_) {
            total += static_cast<std::size_t>(std::popcount(w));
        }
        return total;
    }

    // True iff any bit is set. OR-reduces the words and tests the accumulator once — no
    // per-bit branch.
    [[nodiscard]] auto any() const noexcept -> bool {
        std::uint64_t acc = 0;
        for (const std::uint64_t w : words_) {
            acc |= w;
        }
        return acc != 0;
    }

    [[nodiscard]] auto none() const noexcept -> bool { return !any(); }

    // True iff every index in [0, capacity) is set (vacuously true for capacity 0).
    [[nodiscard]] auto all() const noexcept -> bool { return count() == capacity_; }

    // Index of the lowest set bit, or `capacity` when none is set. Scans words in ascending
    // order and uses std::countr_zero to pluck the bit position inside the first non-zero
    // word (a single hardware CTZ). The only branch is the loop's word-skip, not a per-bit
    // test.
    [[nodiscard]] auto first_set() const noexcept -> std::size_t {
        for (std::size_t wi = 0; wi < words_.size(); ++wi) {
            const std::uint64_t w = words_[wi];
            if (w != 0) {
                return wi * 64 + static_cast<std::size_t>(std::countr_zero(w));
            }
        }
        return capacity_;
    }

    // The indices of all set bits, ascending. Uses the branchless lowest-bit idiom per word:
    // `w & (~w + 1)` isolates the lowest set bit (equivalently -w in two's complement), and
    // std::countr_zero gives its position; `w &= w - 1` then clears it. Each word costs one
    // CTZ per set bit and nothing per clear bit — the classic GPU-friendly bit-scatter.
    [[nodiscard]] auto set_bits() const -> std::vector<std::size_t> {
        std::vector<std::size_t> out;
        out.reserve(count());
        for (std::size_t wi = 0; wi < words_.size(); ++wi) {
            std::uint64_t w = words_[wi];
            while (w != 0) {
                const std::uint64_t low = w & (~w + 1);  // isolate lowest set bit (== w & -w)
                out.push_back(wi * 64 + static_cast<std::size_t>(std::countr_zero(low)));
                w &= w - 1;  // clear the lowest set bit
            }
        }
        return out;
    }

    // Value equality: same capacity and identical bit contents.
    [[nodiscard]] auto operator==(const Bitset& other) const noexcept -> bool {
        return capacity_ == other.capacity_ && words_ == other.words_;
    }

private:
    // The mask of live bits in the final word: all-ones when capacity is a multiple of 64
    // (the last word is fully used), else the low `capacity mod 64` bits. Unused for
    // capacity 0 (there is no last word).
    [[nodiscard]] static auto compute_last_mask(std::size_t capacity) noexcept -> std::uint64_t {
        if (capacity == 0) {
            return 0;
        }
        const std::size_t used = capacity - ((capacity + 63) / 64 - 1) * 64;  // 1..64
        return used == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << used) - 1);
    }

    // Shared body of the Result combinators: capacity-check, then a word-parallel apply of
    // `op` into a fresh bitset. `op` is a plain word->word bitwise lambda (no branches).
    template <typename Op>
    [[nodiscard]] auto combine(const Bitset& other, Op op) const -> Result<Bitset> {
        if (capacity_ != other.capacity_) {
            return make_error<Bitset>(MathError::domain_error);
        }
        Bitset result(capacity_);
        for (std::size_t i = 0; i < words_.size(); ++i) {
            result.words_[i] = op(words_[i], other.words_[i]);
        }
        return result;
    }

    std::size_t capacity_{0};
    std::vector<std::uint64_t> words_{};
    std::uint64_t last_mask_{0};
};

}  // namespace nimblecas
