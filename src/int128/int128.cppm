// NimbleCAS native 128-bit integer & rational (overflow-checked middle tier).
// @author Olumuyiwa Oluwasanmi
//
// A speed tier that sits between the int64 Rational (nimblecas.ratpoly, ~9.2e18 of
// magnitude) and the incoming unbounded arbitrary-precision path
// (nimblecas.bigint / BigRational). It is built on the compiler-native `__int128`,
// which clang-22 lowers to a pair of 64-bit registers on x86-64 — so scalar 128-bit
// add/sub/mul/div stay in registers and are far cheaper than a heap bignum for values
// that fit. `Rational128` mirrors Rational exactly (make(num,den) -> Result, denominator
// kept positive so the numerator sign carries the value's sign, reduced by gcd,
// overflow-checked field ops) but widens the storage to `__int128`.
//
// HONESTY — this tier is STILL BOUNDED. Native signed 128-bit spans roughly
// [-1.7e38, +1.7e38] (exactly [-2^127, 2^127 - 1]). That is ~18 orders of magnitude
// beyond int64, but it has a ceiling. Following the same discipline as the int64
// Rational (Rule 32), an operation whose exact result would leave the __int128 range
// returns MathError::overflow rather than silently wrapping. When a caller sees
// overflow here, the answer is not wrong — it is out of range — and the caller should
// promote the computation to the unbounded nimblecas.bigint / BigRational tier, which
// grows without limit at the cost of heap allocation and slower per-op arithmetic.
//
// NOT VECTORISED. AVX/SIMD offers no scalar 128-bit integer multiply or divide
// instruction — there is no vector op that accelerates a single wide product or
// quotient. SIMD only helps BATCHES of narrower (8/16/32/64-bit) lanes. This module
// is therefore deliberately scalar-native: it wins on latency for one-at-a-time wide
// values, not on throughput of many narrow ones.

export module nimblecas.int128;

import std;
import nimblecas.core;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Int128 — the compiler-native signed 128-bit integer.
// ---------------------------------------------------------------------------
// `__int128` is a first-class arithmetic type on the build target (clang-22,
// x86-64); it simply lacks the standard-library plumbing that `int64_t` gets
// (no std::to_string, no operator<< in <ostream>, no std::numeric_limits guarantees
// portably). The free functions below supply the pieces we actually need.
using Int128 = __int128;

// Decimal rendering of a signed 128-bit value (there is no std::to_string overload).
// Handles the full range including the asymmetric minimum (-2^127), whose magnitude is
// unrepresentable as a positive Int128, by accumulating through unsigned __int128.
[[nodiscard]] auto int128_to_string(Int128 v) -> std::string;

// Parse a base-10 signed integer. Accepts an optional leading '+' or '-' followed by
// one or more decimal digits and nothing else. A non-digit character, a lone sign, or
// an empty string yields MathError::syntax_error; a value outside [-2^127, 2^127 - 1]
// yields MathError::overflow (never a wrapped result).
[[nodiscard]] auto int128_from_string(std::string_view s) -> Result<Int128>;

// Widen a 64-bit integer. Every std::int64_t fits in Int128, so this cannot fail.
[[nodiscard]] auto int128_from_i64(std::int64_t v) noexcept -> Int128;

// Overflow-checked scalar arithmetic. Each detects a 128-bit overflow via the
// corresponding __builtin_*_overflow (which accept __int128 on clang) and returns
// MathError::overflow instead of wrapping.
[[nodiscard]] auto checked_add(Int128 a, Int128 b) -> Result<Int128>;
[[nodiscard]] auto checked_sub(Int128 a, Int128 b) -> Result<Int128>;
[[nodiscard]] auto checked_mul(Int128 a, Int128 b) -> Result<Int128>;

// ---------------------------------------------------------------------------
// Rational128 — an exact fraction num/den in lowest terms with den > 0, over Int128.
// ---------------------------------------------------------------------------
// Field-for-field the wider twin of nimblecas::Rational. The canonical form (den > 0,
// gcd(|num|, den) == 1, zero represented as 0/1) makes equality a member-wise compare
// and keeps intermediate magnitudes as small as the 128-bit ceiling allows. Every
// arithmetic operation is overflow-checked and returns Result; a genuine 128-bit
// overflow surfaces as MathError::overflow so the caller can promote to BigRational.
class Rational128 {
public:
    Rational128() = default;  // 0/1

    // Construct num/den in canonical form. Fails with division_by_zero (den == 0) or
    // overflow (an Int128 boundary such as -2^127 that cannot be sign-normalised).
    [[nodiscard]] static auto make(Int128 num, Int128 den) -> Result<Rational128>;
    [[nodiscard]] static auto from_int(std::int64_t v) -> Rational128;

    [[nodiscard]] auto numerator() const noexcept -> Int128 { return num_; }
    [[nodiscard]] auto denominator() const noexcept -> Int128 { return den_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return num_ == 0; }
    [[nodiscard]] auto is_integer() const noexcept -> bool { return den_ == 1; }

    [[nodiscard]] auto add(const Rational128& o) const -> Result<Rational128>;
    [[nodiscard]] auto subtract(const Rational128& o) const -> Result<Rational128>;
    [[nodiscard]] auto multiply(const Rational128& o) const -> Result<Rational128>;
    // Fails with division_by_zero when o == 0.
    [[nodiscard]] auto divide(const Rational128& o) const -> Result<Rational128>;
    [[nodiscard]] auto negate() const -> Result<Rational128>;      // fails only on -2^127
    // Multiplicative inverse den/num; fails with division_by_zero when *this == 0.
    [[nodiscard]] auto reciprocal() const -> Result<Rational128>;

    [[nodiscard]] auto operator==(const Rational128& o) const noexcept -> bool {
        return num_ == o.num_ && den_ == o.den_;
    }
    [[nodiscard]] auto to_string() const -> std::string;

private:
    Rational128(Int128 num, Int128 den) : num_(num), den_(den) {}
    Int128 num_{0};
    Int128 den_{1};
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// The signed 128-bit bounds. numeric_limits<__int128> is not portably guaranteed, so
// derive them from the two's-complement layout (C++23 mandates two's complement).
inline constexpr Int128 int128_max =
    static_cast<Int128>((static_cast<unsigned __int128>(1) << 127) - 1);   //  2^127 - 1
inline constexpr Int128 int128_min = -int128_max - 1;                      // -2^127

// Euclidean gcd on Int128, returning a non-negative result; gcd(0, 0) == 0. The
// callers guarantee neither argument is int128_min, so the -v magnitude flips below are
// always representable.
[[nodiscard]] auto int128_gcd(Int128 a, Int128 b) -> Int128 {
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
    while (b != 0) {
        const Int128 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

}  // namespace

// --- Int128 free functions ---------------------------------------------------

auto int128_to_string(Int128 v) -> std::string {
    if (v == 0) {
        return "0";
    }
    const bool neg = v < 0;
    // Accumulate the magnitude in unsigned space: 0u - (unsigned)v yields |v| even for
    // v == int128_min (2^127), which has no positive signed representation.
    unsigned __int128 u = neg ? static_cast<unsigned __int128>(0) - static_cast<unsigned __int128>(v)
                              : static_cast<unsigned __int128>(v);
    std::string s;
    while (u != 0) {
        s.push_back(static_cast<char>('0' + static_cast<int>(u % 10)));
        u /= 10;
    }
    if (neg) {
        s.push_back('-');
    }
    std::ranges::reverse(s);
    return s;
}

auto int128_from_string(std::string_view s) -> Result<Int128> {
    if (s.empty()) {
        return make_error<Int128>(MathError::syntax_error);
    }
    std::size_t i = 0;
    bool neg = false;
    if (s[i] == '+' || s[i] == '-') {
        neg = (s[i] == '-');
        ++i;
    }
    if (i >= s.size()) {  // a lone sign with no digits
        return make_error<Int128>(MathError::syntax_error);
    }
    // Bound the accumulated magnitude: |int128_min| = 2^127 for a negative literal,
    // int128_max = 2^127 - 1 for a non-negative one.
    const unsigned __int128 max_mag =
        neg ? (static_cast<unsigned __int128>(1) << 127)
            : ((static_cast<unsigned __int128>(1) << 127) - 1);
    unsigned __int128 acc = 0;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c < '0' || c > '9') {
            return make_error<Int128>(MathError::syntax_error);
        }
        const unsigned d = static_cast<unsigned>(c - '0');
        // acc*10 + d must stay <= max_mag; check before forming it so nothing wraps.
        if (acc > (max_mag - d) / 10) {
            return make_error<Int128>(MathError::overflow);
        }
        acc = acc * 10 + d;
    }
    // Reinterpret the magnitude with sign. For neg, 0u - acc gives the two's-complement
    // bit pattern of -acc, correct even when acc == 2^127 (yielding int128_min).
    return neg ? static_cast<Int128>(static_cast<unsigned __int128>(0) - acc)
               : static_cast<Int128>(acc);
}

auto int128_from_i64(std::int64_t v) noexcept -> Int128 {
    return static_cast<Int128>(v);
}

auto checked_add(Int128 a, Int128 b) -> Result<Int128> {
    Int128 out = 0;
    if (__builtin_add_overflow(a, b, &out)) {
        return make_error<Int128>(MathError::overflow);
    }
    return out;
}

auto checked_sub(Int128 a, Int128 b) -> Result<Int128> {
    Int128 out = 0;
    if (__builtin_sub_overflow(a, b, &out)) {
        return make_error<Int128>(MathError::overflow);
    }
    return out;
}

auto checked_mul(Int128 a, Int128 b) -> Result<Int128> {
    Int128 out = 0;
    if (__builtin_mul_overflow(a, b, &out)) {
        return make_error<Int128>(MathError::overflow);
    }
    return out;
}

// --- Rational128 -------------------------------------------------------------

auto Rational128::make(Int128 num, Int128 den) -> Result<Rational128> {
    if (den == 0) {
        return make_error<Rational128>(MathError::division_by_zero);
    }
    // Normalise the sign onto the numerator so the denominator stays positive. Negating
    // int128_min overflows, so reject either operand at that boundary up front — after
    // this guard every magnitude flip and gcd step is representable.
    if (num == int128_min || den == int128_min) {
        return make_error<Rational128>(MathError::overflow);
    }
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const Int128 g = int128_gcd(num, den);  // g >= 1 here (den > 0)
    return Rational128{num / g, den / g};
}

auto Rational128::from_int(std::int64_t v) -> Rational128 {
    return Rational128{int128_from_i64(v), 1};
}

auto Rational128::add(const Rational128& o) const -> Result<Rational128> {
    // num_/den_ + o.num_/o.den_ = (num_*o.den_ + o.num_*den_) / (den_*o.den_).
    auto t1 = checked_mul(num_, o.den_);
    if (!t1) {
        return make_error<Rational128>(t1.error());
    }
    auto t2 = checked_mul(o.num_, den_);
    if (!t2) {
        return make_error<Rational128>(t2.error());
    }
    auto num = checked_add(*t1, *t2);
    if (!num) {
        return make_error<Rational128>(num.error());
    }
    auto den = checked_mul(den_, o.den_);
    if (!den) {
        return make_error<Rational128>(den.error());
    }
    return make(*num, *den);
}

auto Rational128::subtract(const Rational128& o) const -> Result<Rational128> {
    auto neg = o.negate();
    if (!neg) {
        return neg;
    }
    return add(*neg);
}

auto Rational128::multiply(const Rational128& o) const -> Result<Rational128> {
    auto num = checked_mul(num_, o.num_);
    if (!num) {
        return make_error<Rational128>(num.error());
    }
    auto den = checked_mul(den_, o.den_);
    if (!den) {
        return make_error<Rational128>(den.error());
    }
    return make(*num, *den);
}

auto Rational128::divide(const Rational128& o) const -> Result<Rational128> {
    if (o.num_ == 0) {
        return make_error<Rational128>(MathError::division_by_zero);
    }
    // (num_/den_) / (o.num_/o.den_) = (num_*o.den_) / (den_*o.num_).
    auto num = checked_mul(num_, o.den_);
    if (!num) {
        return make_error<Rational128>(num.error());
    }
    auto den = checked_mul(den_, o.num_);
    if (!den) {
        return make_error<Rational128>(den.error());
    }
    return make(*num, *den);  // make() re-normalises the sign when o.num_ < 0
}

auto Rational128::negate() const -> Result<Rational128> {
    if (num_ == int128_min) {
        return make_error<Rational128>(MathError::overflow);
    }
    return Rational128{-num_, den_};  // already canonical (den_ > 0, gcd unchanged)
}

auto Rational128::reciprocal() const -> Result<Rational128> {
    if (num_ == 0) {
        return make_error<Rational128>(MathError::division_by_zero);
    }
    // 1 / (num_/den_) = den_/num_. den_ is positive and num_ is never int128_min
    // (make() rejects that operand), so make() cannot hit its overflow guard here.
    return make(den_, num_);
}

auto Rational128::to_string() const -> std::string {
    return den_ == 1 ? int128_to_string(num_)
                     : int128_to_string(num_) + "/" + int128_to_string(den_);
}

}  // namespace nimblecas
