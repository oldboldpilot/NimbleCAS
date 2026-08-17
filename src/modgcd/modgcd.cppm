// NimbleCAS distributed modular polynomial GCD math core (Brown's modular GCD for Z[x]).
// @author Olumuyiwa Oluwasanmi
//
// Brown's modular GCD algorithm for univariate polynomials in Z[x] with exact int64
// coefficients: computes the primitive-part GCD by reducing modulo single-word primes
// p in (2^30, 2^31), running monic Euclidean GCD over Z_p[x], lifting the modular
// images via symmetric Chinese Remainder Theorem (CRT), and strictly verifying the
// candidate by exact trial division (Polynomial::divide_exact) before returning.
//
// HONESTY BOUNDARY:
// - Exact and complete: returns the exact polynomial GCD or an honest MathError.
// - Returns MathError::not_converged if the prime budget is exhausted without
//   producing a trial-division-verified candidate.
// - Returns MathError::overflow if reconstructed coefficients or content scaling
//   exceed the int64 range.
// - NEVER returns an unverified or approximate candidate.

export module nimblecas.modgcd;

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.bigint;
import nimblecas.numbertheory;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Types & Constants
// ---------------------------------------------------------------------------

// Modular image of a GCD in Z_p[x] (canonical residues in [0, p), gamma-scaled monic).
struct ZpImage {
    std::uint64_t p{0};
    std::vector<std::uint64_t> coeffs{};

    [[nodiscard]] auto operator==(const ZpImage& other) const noexcept -> bool = default;
};

// Outcomes of merge_images.
struct NeedMorePrimes {
    [[nodiscard]] auto operator==(const NeedMorePrimes&) const noexcept -> bool = default;
};

struct CoprimeProven {
    [[nodiscard]] auto operator==(const CoprimeProven&) const noexcept -> bool = default;
};

struct Candidate {
    Polynomial polynomial{};

    [[nodiscard]] auto operator==(const Candidate& other) const noexcept -> bool {
        return polynomial.is_equal(other.polynomial);
    }
};

using MergeOutcome = std::variant<NeedMorePrimes, CoprimeProven, Candidate>;

// Request payload for worker image computation.
struct ImageRequest {
    std::uint64_t p{0};
    std::int64_t gamma{0};
    Polynomial a{};
    Polynomial b{};

    [[nodiscard]] auto operator==(const ImageRequest& other) const noexcept -> bool {
        return p == other.p && gamma == other.gamma &&
               a.is_equal(other.a) && b.is_equal(other.b);
    }
};

using ImageResult = ZpImage;

inline constexpr std::uint32_t k_image_request_magic = 0x4E434751;  // "NCGQ" in LE
inline constexpr std::uint32_t k_image_result_magic  = 0x4E434749;  // "NCGI" in LE
inline constexpr std::size_t k_max_payload_size = 64 * 1024 * 1024; // 64 MiB
inline constexpr std::size_t k_default_prime_budget = 256;

// ---------------------------------------------------------------------------
// Exported Codec API
// ---------------------------------------------------------------------------

// Polynomial inner block codec (embedded inside ImageRequest).
[[nodiscard]] auto encode_polynomial(const Polynomial& poly, std::vector<std::byte>& out)
    -> Result<void>;
[[nodiscard]] auto encode_polynomial(const Polynomial& poly) -> Result<std::vector<std::byte>>;
[[nodiscard]] auto decode_polynomial(std::span<const std::byte> bytes, std::size_t& offset)
    -> Result<Polynomial>;
[[nodiscard]] auto decode_polynomial(std::span<const std::byte> bytes) -> Result<Polynomial>;

// ImageRequest codec.
[[nodiscard]] auto encode_image_request(const ImageRequest& req)
    -> Result<std::vector<std::byte>>;
[[nodiscard]] auto decode_image_request(std::span<const std::byte> bytes)
    -> Result<ImageRequest>;

// ImageResult codec.
[[nodiscard]] auto encode_image_result(const ZpImage& res)
    -> Result<std::vector<std::byte>>;
[[nodiscard]] auto decode_image_result(std::span<const std::byte> bytes)
    -> Result<ZpImage>;

// ---------------------------------------------------------------------------
// Exported Math Core API
// ---------------------------------------------------------------------------

// Landau-Mignotte coefficient bound in BigInt (sqrt-free formulation).
[[nodiscard]] auto landau_mignotte_bound(const Polynomial& A, const Polynomial& B)
    -> Result<BigInt>;

// Per-prime image kernel over Z_p[x].
[[nodiscard]] auto gcd_image_mod_p(const Polynomial& A, const Polynomial& B,
                                   std::uint64_t p, std::int64_t gamma)
    -> Result<ZpImage>;

// Symmetric CRT lift and merge over accumulated modular images.
[[nodiscard]] auto merge_images(std::span<const ZpImage> images, std::int64_t gamma,
                                const BigInt& b_lm) -> Result<MergeOutcome>;

// In-process reference driver for modular GCD over Z[x].
[[nodiscard]] auto modular_gcd(const Polynomial& a, const Polynomial& b,
                               std::size_t max_primes = k_default_prime_budget)
    -> Result<Polynomial>;

}  // namespace nimblecas

// ===========================================================================
// Implementation
// ===========================================================================
namespace nimblecas {
namespace {

inline auto write_u16_le(std::uint16_t val, std::vector<std::byte>& out) -> void {
    out.push_back(static_cast<std::byte>(val & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 8) & 0xFF));
}

inline auto write_u32_le(std::uint32_t val, std::vector<std::byte>& out) -> void {
    out.push_back(static_cast<std::byte>(val & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 8) & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 16) & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 24) & 0xFF));
}

inline auto write_u64_le(std::uint64_t val, std::vector<std::byte>& out) -> void {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((val >> (i * 8)) & 0xFF));
    }
}

inline auto read_u16_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint16_t {
    return static_cast<std::uint16_t>(bytes[offset]) |
           (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

inline auto read_u32_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint32_t {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

inline auto read_u64_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint64_t {
    std::uint64_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val |= (static_cast<std::uint64_t>(bytes[offset + i]) << (i * 8));
    }
    return val;
}

// std::gcd guard for INT64_MIN
[[nodiscard]] auto checked_gcd(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (a == int64_min || b == int64_min) {
        return std::nullopt;
    }
    return std::gcd(a, b);
}

// Convert BigInt to std::int64_t
[[nodiscard]] auto bigint_to_i64(const BigInt& x) -> std::optional<std::int64_t> {
    const std::string s = x.to_string();
    std::int64_t out = 0;
    const auto* end = s.data() + s.size();
    const auto res = std::from_chars(s.data(), end, out);
    if (res.ec != std::errc{} || res.ptr != end) {
        return std::nullopt;
    }
    return out;
}

// Convert BigInt to std::uint64_t
[[nodiscard]] auto bigint_to_u64(const BigInt& x) -> std::optional<std::uint64_t> {
    if (x.is_negative()) {
        return std::nullopt;
    }
    const std::string s = x.to_string();
    std::uint64_t out = 0;
    const auto* end = s.data() + s.size();
    const auto res = std::from_chars(s.data(), end, out);
    if (res.ec != std::errc{} || res.ptr != end) {
        return std::nullopt;
    }
    return out;
}

// Modular multiplication with unsigned __int128 to guard against overflow
[[nodiscard]] auto mod_mul(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept -> std::uint64_t {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % p);
}

// Modular exponentiation
[[nodiscard]] auto mod_pow(std::uint64_t base, std::uint64_t exp, std::uint64_t p) noexcept -> std::uint64_t {
    std::uint64_t res = 1 % p;
    base %= p;
    while (exp > 0) {
        if (exp & 1) {
            res = mod_mul(res, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    return res;
}

// Modular inverse modulo a prime p via Fermat's Little Theorem
[[nodiscard]] auto mod_inv_prime(std::uint64_t a, std::uint64_t p) noexcept -> std::uint64_t {
    return mod_pow(a, p - 2, p);
}

// Map an int64_t into canonical residue [0, p)
[[nodiscard]] auto to_zp_residue(std::int64_t c, std::uint64_t p) noexcept -> std::uint64_t {
    if (c < 0) {
        const std::uint64_t mag = 0ULL - static_cast<std::uint64_t>(c);
        const std::uint64_t r = mag % p;
        return (r == 0) ? 0ULL : (p - r);
    }
    return static_cast<std::uint64_t>(c) % p;
}

// Trim trailing zeros from a Z_p polynomial
auto trim_zp(std::vector<std::uint64_t>& coeffs) noexcept -> void {
    while (!coeffs.empty() && coeffs.back() == 0) {
        coeffs.pop_back();
    }
}

// Polynomial remainder in Z_p[x]: a mod b
[[nodiscard]] auto zp_poly_rem(std::vector<std::uint64_t> a,
                               const std::vector<std::uint64_t>& b,
                               std::uint64_t p) -> std::vector<std::uint64_t> {
    trim_zp(a);
    if (b.empty()) {
        return a;
    }
    const std::size_t deg_b = b.size() - 1;
    const std::uint64_t lc_b = b.back();
    const std::uint64_t inv_lc_b = mod_inv_prime(lc_b, p);

    while (!a.empty() && a.size() - 1 >= deg_b) {
        const std::size_t deg_a = a.size() - 1;
        const std::size_t shift = deg_a - deg_b;
        const std::uint64_t factor = mod_mul(a.back(), inv_lc_b, p);

        for (std::size_t i = 0; i <= deg_b; ++i) {
            const std::uint64_t sub = mod_mul(b[i], factor, p);
            const std::uint64_t cur = a[i + shift];
            a[i + shift] = (cur >= sub) ? (cur - sub) : (cur + p - sub);
        }
        trim_zp(a);
    }
    return a;
}

// Prime schedule generator: q_0 = 2^30, q_{k+1} = next_prime(q_k, 0), skipping primes dividing lc(A) or lc(B)
class PrimeSchedule {
public:
    explicit PrimeSchedule(std::int64_t lc_a, std::int64_t lc_b)
        : current_(BigInt::from_u64(1ULL << 30)) {
        u_lc_a_ = (lc_a < 0) ? (0ULL - static_cast<std::uint64_t>(lc_a))
                             : static_cast<std::uint64_t>(lc_a);
        u_lc_b_ = (lc_b < 0) ? (0ULL - static_cast<std::uint64_t>(lc_b))
                             : static_cast<std::uint64_t>(lc_b);
    }

    [[nodiscard]] auto next() -> Result<std::uint64_t> {
        while (true) {
            auto next_res = next_prime(current_, 0);
            if (!next_res) {
                return make_error<std::uint64_t>(next_res.error());
            }
            current_ = *next_res;
            auto p_opt = bigint_to_u64(current_);
            if (!p_opt) {
                return make_error<std::uint64_t>(MathError::overflow);
            }
            const std::uint64_t p = *p_opt;
            if (u_lc_a_ > 0 && (u_lc_a_ % p == 0)) {
                continue;
            }
            if (u_lc_b_ > 0 && (u_lc_b_ % p == 0)) {
                continue;
            }
            return p;
        }
    }

private:
    std::uint64_t u_lc_a_{0};
    std::uint64_t u_lc_b_{0};
    BigInt current_{};
};

}  // namespace

// ---------------------------------------------------------------------------
// Polynomial Inner Block Codec
// ---------------------------------------------------------------------------

auto encode_polynomial(const Polynomial& poly, std::vector<std::byte>& out)
    -> Result<void> {
    const auto coeffs = poly.coefficients();
    const std::size_t n = coeffs.size();
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        return make_error<void>(MathError::overflow);
    }
    const std::size_t added_size = 4 + n * 8;
    if (out.size() + added_size > k_max_payload_size) {
        return make_error<void>(MathError::overflow);
    }

    write_u32_le(static_cast<std::uint32_t>(n), out);
    for (const std::int64_t c : coeffs) {
        write_u64_le(std::bit_cast<std::uint64_t>(c), out);
    }
    return {};
}

auto encode_polynomial(const Polynomial& poly) -> Result<std::vector<std::byte>> {
    std::vector<std::byte> out;
    auto res = encode_polynomial(poly, out);
    if (!res) {
        return make_error<std::vector<std::byte>>(res.error());
    }
    return out;
}

auto decode_polynomial(std::span<const std::byte> bytes, std::size_t& offset)
    -> Result<Polynomial> {
    if (offset + 4 > bytes.size()) {
        return make_error<Polynomial>(MathError::syntax_error);
    }
    const std::uint32_t n = read_u32_le(bytes, offset);
    offset += 4;

    if (n == 0) {
        return Polynomial{};
    }

    if (static_cast<std::size_t>(n) > (bytes.size() - offset) / 8) {
        return make_error<Polynomial>(MathError::syntax_error);
    }

    std::vector<std::int64_t> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t u = read_u64_le(bytes, offset);
        offset += 8;
        coeffs[i] = std::bit_cast<std::int64_t>(u);
    }

    // A nonzero count whose last coefficient is 0 is untrimmed and non-canonical
    if (coeffs.back() == 0) {
        return make_error<Polynomial>(MathError::syntax_error);
    }

    return Polynomial{std::move(coeffs)};
}

auto decode_polynomial(std::span<const std::byte> bytes) -> Result<Polynomial> {
    std::size_t offset = 0;
    auto res = decode_polynomial(bytes, offset);
    if (!res) {
        return res;
    }
    if (offset != bytes.size()) {
        return make_error<Polynomial>(MathError::syntax_error);
    }
    return res;
}

// ---------------------------------------------------------------------------
// ImageRequest Codec
// ---------------------------------------------------------------------------

auto encode_image_request(const ImageRequest& req) -> Result<std::vector<std::byte>> {
    const std::size_t na = req.a.coefficients().size();
    const std::size_t nb = req.b.coefficients().size();
    const std::size_t total_size = 4 + 2 + 2 + 8 + 8 + 4 + na * 8 + 4 + nb * 8;
    if (total_size > k_max_payload_size) {
        return make_error<std::vector<std::byte>>(MathError::overflow);
    }

    std::vector<std::byte> out;
    out.reserve(total_size);

    write_u32_le(k_image_request_magic, out);
    write_u16_le(1, out);  // version 1
    write_u16_le(0, out);  // reserved
    write_u64_le(req.p, out);
    write_u64_le(std::bit_cast<std::uint64_t>(req.gamma), out);

    auto ra = encode_polynomial(req.a, out);
    if (!ra) {
        return make_error<std::vector<std::byte>>(ra.error());
    }
    auto rb = encode_polynomial(req.b, out);
    if (!rb) {
        return make_error<std::vector<std::byte>>(rb.error());
    }

    return out;
}

auto decode_image_request(std::span<const std::byte> bytes) -> Result<ImageRequest> {
    if (bytes.size() < 24) {
        return make_error<ImageRequest>(MathError::syntax_error);
    }

    const std::uint32_t magic = read_u32_le(bytes, 0);
    if (magic != k_image_request_magic) {
        return make_error<ImageRequest>(MathError::syntax_error);
    }

    const std::uint16_t version = read_u16_le(bytes, 4);
    if (version != 1) {
        return make_error<ImageRequest>(MathError::syntax_error);
    }

    const std::uint64_t p = read_u64_le(bytes, 8);
    const std::uint64_t gamma_u = read_u64_le(bytes, 16);
    const std::int64_t gamma = std::bit_cast<std::int64_t>(gamma_u);

    std::size_t offset = 24;
    auto a_res = decode_polynomial(bytes, offset);
    if (!a_res) {
        return make_error<ImageRequest>(a_res.error());
    }
    auto b_res = decode_polynomial(bytes, offset);
    if (!b_res) {
        return make_error<ImageRequest>(b_res.error());
    }

    if (offset != bytes.size()) {
        return make_error<ImageRequest>(MathError::syntax_error);
    }

    // Validation checks per spec section 5.2
    if (p <= (1ULL << 30) || p >= (1ULL << 31)) {
        return make_error<ImageRequest>(MathError::domain_error);
    }
    if (gamma <= 0) {
        return make_error<ImageRequest>(MathError::domain_error);
    }
    if (a_res->is_zero() || a_res->degree() <= 0 || b_res->is_zero() || b_res->degree() <= 0) {
        return make_error<ImageRequest>(MathError::domain_error);
    }
    if (to_zp_residue(a_res->leading_coefficient(), p) == 0 ||
        to_zp_residue(b_res->leading_coefficient(), p) == 0) {
        return make_error<ImageRequest>(MathError::domain_error);
    }

    return ImageRequest{
        .p = p,
        .gamma = gamma,
        .a = std::move(*a_res),
        .b = std::move(*b_res),
    };
}

// ---------------------------------------------------------------------------
// ImageResult Codec
// ---------------------------------------------------------------------------

auto encode_image_result(const ZpImage& res) -> Result<std::vector<std::byte>> {
    if (res.coeffs.empty()) {
        return make_error<std::vector<std::byte>>(MathError::domain_error);
    }
    const std::size_t count = res.coeffs.size();
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        return make_error<std::vector<std::byte>>(MathError::overflow);
    }
    const std::size_t total_size = 4 + 2 + 2 + 8 + 4 + count * 8;
    if (total_size > k_max_payload_size) {
        return make_error<std::vector<std::byte>>(MathError::overflow);
    }

    std::vector<std::byte> out;
    out.reserve(total_size);

    write_u32_le(k_image_result_magic, out);
    write_u16_le(1, out);  // version 1
    write_u16_le(0, out);  // reserved
    write_u64_le(res.p, out);
    write_u32_le(static_cast<std::uint32_t>(count), out);

    for (const std::uint64_t c : res.coeffs) {
        write_u64_le(c, out);
    }
    return out;
}

auto decode_image_result(std::span<const std::byte> bytes) -> Result<ZpImage> {
    if (bytes.size() < 20) {
        return make_error<ZpImage>(MathError::syntax_error);
    }

    const std::uint32_t magic = read_u32_le(bytes, 0);
    if (magic != k_image_result_magic) {
        return make_error<ZpImage>(MathError::syntax_error);
    }

    const std::uint16_t version = read_u16_le(bytes, 4);
    if (version != 1) {
        return make_error<ZpImage>(MathError::syntax_error);
    }

    const std::uint64_t p = read_u64_le(bytes, 8);
    const std::uint32_t count = read_u32_le(bytes, 16);

    if (count == 0) {
        return make_error<ZpImage>(MathError::syntax_error);
    }

    const std::size_t expected_size = 20 + static_cast<std::size_t>(count) * 8;
    if (bytes.size() != expected_size) {
        return make_error<ZpImage>(MathError::syntax_error);
    }

    std::vector<std::uint64_t> coeffs(count);
    std::size_t offset = 20;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t val = read_u64_le(bytes, offset);
        offset += 8;
        if (val >= p) {
            return make_error<ZpImage>(MathError::domain_error);
        }
        coeffs[i] = val;
    }

    if (count > 1 && coeffs.back() == 0) {
        return make_error<ZpImage>(MathError::domain_error);
    }

    return ZpImage{
        .p = p,
        .coeffs = std::move(coeffs),
    };
}

// ---------------------------------------------------------------------------
// Landau-Mignotte Bound
// ---------------------------------------------------------------------------

auto landau_mignotte_bound(const Polynomial& A, const Polynomial& B)
    -> Result<BigInt> {
    if (A.is_zero() || B.is_zero() || A.degree() < 0 || B.degree() < 0) {
        return make_error<BigInt>(MathError::domain_error);
    }

    const std::int64_t dA = A.degree();
    const std::int64_t dB = B.degree();

    auto gamma_opt = checked_gcd(A.leading_coefficient(), B.leading_coefficient());
    if (!gamma_opt || *gamma_opt <= 0) {
        return make_error<BigInt>(MathError::domain_error);
    }
    const std::int64_t gamma = *gamma_opt;

    // Find infinity norms ‖A‖∞ and ‖B‖∞
    std::uint64_t max_a = 0;
    for (const std::int64_t c : A.coefficients()) {
        const std::uint64_t mag = (c < 0) ? (0ULL - static_cast<std::uint64_t>(c))
                                          : static_cast<std::uint64_t>(c);
        max_a = std::max(max_a, mag);
    }

    std::uint64_t max_b = 0;
    for (const std::int64_t c : B.coefficients()) {
        const std::uint64_t mag = (c < 0) ? (0ULL - static_cast<std::uint64_t>(c))
                                          : static_cast<std::uint64_t>(c);
        max_b = std::max(max_b, mag);
    }

    const BigInt term_A = BigInt::from_u64(static_cast<std::uint64_t>(dA + 1))
                              .multiply(BigInt::from_u64(max_a));
    const BigInt term_B = BigInt::from_u64(static_cast<std::uint64_t>(dB + 1))
                              .multiply(BigInt::from_u64(max_b));
    const BigInt min_term = (term_A < term_B) ? term_A : term_B;

    const std::uint64_t min_deg = static_cast<std::uint64_t>(std::min(dA, dB));
    const BigInt pow2 = BigInt::from_u64(2).pow(min_deg);
    const BigInt gamma_bi = BigInt::from_u64(static_cast<std::uint64_t>(gamma));

    return gamma_bi.multiply(pow2).multiply(min_term);
}

// ---------------------------------------------------------------------------
// GCD Image Modulo p Kernel
// ---------------------------------------------------------------------------

auto gcd_image_mod_p(const Polynomial& A, const Polynomial& B,
                     std::uint64_t p, std::int64_t gamma)
    -> Result<ZpImage> {
    if (p < 2 || gamma <= 0 || A.is_zero() || B.is_zero()) {
        return make_error<ZpImage>(MathError::domain_error);
    }

    const std::int64_t lcA = A.leading_coefficient();
    const std::int64_t lcB = B.leading_coefficient();
    if (to_zp_residue(lcA, p) == 0 || to_zp_residue(lcB, p) == 0) {
        return make_error<ZpImage>(MathError::domain_error);
    }

    std::vector<std::uint64_t> pa(A.coefficients().size());
    for (std::size_t i = 0; i < A.coefficients().size(); ++i) {
        pa[i] = to_zp_residue(A.coefficients()[i], p);
    }
    trim_zp(pa);

    std::vector<std::uint64_t> pb(B.coefficients().size());
    for (std::size_t i = 0; i < B.coefficients().size(); ++i) {
        pb[i] = to_zp_residue(B.coefficients()[i], p);
    }
    trim_zp(pb);

    while (!pb.empty()) {
        auto r = zp_poly_rem(std::move(pa), pb, p);
        pa = std::move(pb);
        pb = std::move(r);
    }

    if (pa.empty()) {
        return ZpImage{.p = p, .coeffs = {}};
    }

    // Make monic
    const std::uint64_t lc = pa.back();
    const std::uint64_t inv_lc = mod_inv_prime(lc, p);
    for (auto& c : pa) {
        c = mod_mul(c, inv_lc, p);
    }

    // Multiply by gamma mod p
    const std::uint64_t gamma_mod = to_zp_residue(gamma, p);
    for (auto& c : pa) {
        c = mod_mul(c, gamma_mod, p);
    }

    trim_zp(pa);
    return ZpImage{.p = p, .coeffs = std::move(pa)};
}

// ---------------------------------------------------------------------------
// Merge Images (Symmetric CRT Lift)
// ---------------------------------------------------------------------------

auto merge_images(std::span<const ZpImage> images, std::int64_t gamma,
                  const BigInt& b_lm) -> Result<MergeOutcome> {
    if (images.empty()) {
        return MergeOutcome{NeedMorePrimes{}};
    }
    if (gamma <= 0) {
        return make_error<MergeOutcome>(MathError::domain_error);
    }

    // Find d_min over all images seen so far
    std::size_t d_min = std::numeric_limits<std::size_t>::max();
    for (const auto& img : images) {
        if (img.coeffs.empty()) {
            return make_error<MergeOutcome>(MathError::domain_error);
        }
        const std::size_t deg = img.coeffs.size() - 1;
        if (deg < d_min) {
            d_min = deg;
        }
    }

    // Degree 0 image bounds deg(g) <= 0, proving primitive parts are coprime
    if (d_min == 0) {
        return MergeOutcome{CoprimeProven{}};
    }

    // Accept images with degree == d_min in prime-index order
    std::vector<const ZpImage*> accepted;
    accepted.reserve(images.size());
    BigInt M = BigInt::from_u64(1);
    for (const auto& img : images) {
        if (img.coeffs.size() - 1 == d_min) {
            accepted.push_back(&img);
            M = M.multiply(BigInt::from_u64(img.p));
        }
    }

    const BigInt req_M = BigInt::from_u64(2).multiply(b_lm);
    if (!(M > req_M)) {
        return MergeOutcome{NeedMorePrimes{}};
    }

    // Use the minimal prefix of accepted images whose modulus product exceeds 2*B_lm
    std::vector<const ZpImage*> active_accepted;
    BigInt active_M = BigInt::from_u64(1);
    for (const auto* img : accepted) {
        active_accepted.push_back(img);
        active_M = active_M.multiply(BigInt::from_u64(img->p));
        if (active_M > req_M) {
            break;
        }
    }

    const std::size_t num_coeffs = d_min + 1;
    std::vector<BigInt> moduli;
    moduli.reserve(active_accepted.size());
    for (const auto* img : active_accepted) {
        moduli.push_back(BigInt::from_u64(img->p));
    }

    const BigInt one = BigInt::from_u64(1);
    const BigInt two = BigInt::from_u64(2);
    auto half_m_res = active_M.subtract(one).divmod(two);
    if (!half_m_res) {
        return make_error<MergeOutcome>(half_m_res.error());
    }
    const BigInt half_M = half_m_res->first;

    std::vector<BigInt> lifted_coeffs(num_coeffs);
    for (std::size_t j = 0; j < num_coeffs; ++j) {
        std::vector<BigInt> residues;
        residues.reserve(active_accepted.size());
        for (const auto* img : active_accepted) {
            residues.push_back(BigInt::from_u64(img->coeffs[j]));
        }
        auto crt_res = crt(residues, moduli);
        if (!crt_res) {
            return make_error<MergeOutcome>(crt_res.error());
        }
        BigInt xj = *crt_res;
        // Symmetric lift
        if (xj > half_M) {
            lifted_coeffs[j] = xj.subtract(active_M);
        } else {
            lifted_coeffs[j] = std::move(xj);
        }
    }

    // Leading coefficient sanity check: must equal gamma exactly
    const BigInt gamma_bi = BigInt::from_i64(gamma);
    if (lifted_coeffs.back() != gamma_bi) {
        return make_error<MergeOutcome>(MathError::domain_error);
    }

    // Primitive part in BigInt
    BigInt content_bi = BigInt::from_u64(0);
    for (const auto& c : lifted_coeffs) {
        content_bi = BigInt::gcd(content_bi, c.abs());
    }
    if (content_bi.is_zero()) {
        return make_error<MergeOutcome>(MathError::domain_error);
    }

    for (auto& c : lifted_coeffs) {
        auto div_res = c.divmod(content_bi);
        if (!div_res || !div_res->second.is_zero()) {
            return make_error<MergeOutcome>(MathError::domain_error);
        }
        c = div_res->first;
    }

    // Sign-normalise primitive part to positive leading coefficient
    if (lifted_coeffs.back().is_negative()) {
        for (auto& c : lifted_coeffs) {
            c = c.negate();
        }
    }

    // Narrow to int64
    std::vector<std::int64_t> result_coeffs(num_coeffs);
    for (std::size_t j = 0; j < num_coeffs; ++j) {
        auto i64_opt = bigint_to_i64(lifted_coeffs[j]);
        if (!i64_opt) {
            return make_error<MergeOutcome>(MathError::overflow);
        }
        result_coeffs[j] = *i64_opt;
    }

    return MergeOutcome{Candidate{Polynomial{std::move(result_coeffs)}}};
}

// ---------------------------------------------------------------------------
// In-process Reference Driver
// ---------------------------------------------------------------------------

auto modular_gcd(const Polynomial& a, const Polynomial& b, std::size_t max_primes)
    -> Result<Polynomial> {
    auto normalized = [](const Polynomial& p) -> Result<Polynomial> {
        return p.leading_coefficient() < 0 ? p.scale(-1) : Result<Polynomial>{p};
    };
    if (a.is_zero()) {
        return normalized(b);
    }
    if (b.is_zero()) {
        return normalized(a);
    }

    // Content split
    auto ca = a.content();
    auto cb = b.content();
    if (!ca || !cb) {
        return make_error<Polynomial>(MathError::overflow);
    }
    auto d_opt = checked_gcd(*ca, *cb);
    if (!d_opt) {
        return make_error<Polynomial>(MathError::overflow);
    }
    const std::int64_t d = *d_opt;

    auto a_prim = a.primitive_part();
    auto b_prim = b.primitive_part();
    if (!a_prim || !b_prim) {
        return make_error<Polynomial>(a_prim ? b_prim.error() : a_prim.error());
    }
    const Polynomial A = *a_prim;
    const Polynomial B = *b_prim;

    // Constant primitive parts
    if (A.degree() == 0 || B.degree() == 0) {
        return Polynomial::constant(d);
    }

    // gamma = gcd(lc(A), lc(B))
    auto gamma_opt = checked_gcd(A.leading_coefficient(), B.leading_coefficient());
    if (!gamma_opt) {
        return make_error<Polynomial>(MathError::overflow);
    }
    const std::int64_t gamma = *gamma_opt;

    // Coefficient bound
    auto b_lm_res = landau_mignotte_bound(A, B);
    if (!b_lm_res) {
        return make_error<Polynomial>(b_lm_res.error());
    }
    const BigInt b_lm = *b_lm_res;

    // Sizing: least k with 2^k > 2 * B_lm
    const BigInt target_prod = BigInt::from_u64(2).multiply(b_lm);
    std::size_t k = 1;
    BigInt pow2 = BigInt::from_u64(2);
    while (!(pow2 > target_prod)) {
        pow2 = pow2.multiply(BigInt::from_u64(2));
        ++k;
    }
    const std::size_t m_target = std::max<std::size_t>(1, (k + 29) / 30);

    PrimeSchedule sched(A.leading_coefficient(), B.leading_coefficient());
    std::vector<ZpImage> all_images;
    std::size_t primes_consumed = 0;
    bool first_round = true;

    while (primes_consumed < max_primes) {
        std::size_t m_request = 0;
        if (first_round) {
            m_request = m_target + 2;
            first_round = false;
        } else {
            // Count accepted images with minimal degree so far
            std::size_t d_min = std::numeric_limits<std::size_t>::max();
            for (const auto& img : all_images) {
                if (!img.coeffs.empty() && img.coeffs.size() - 1 < d_min) {
                    d_min = img.coeffs.size() - 1;
                }
            }
            std::size_t m_accepted = 0;
            for (const auto& img : all_images) {
                if (img.coeffs.size() - 1 == d_min) {
                    ++m_accepted;
                }
            }
            const std::int64_t diff = static_cast<std::int64_t>(m_target) -
                                      static_cast<std::int64_t>(m_accepted);
            m_request = static_cast<std::size_t>(std::max<std::int64_t>(diff, 2) + 2);
        }

        // Cap m_request by remaining prime budget
        if (primes_consumed + m_request > max_primes) {
            m_request = max_primes - primes_consumed;
        }
        if (m_request == 0) {
            break;
        }

        for (std::size_t i = 0; i < m_request; ++i) {
            auto p_res = sched.next();
            if (!p_res) {
                return make_error<Polynomial>(p_res.error());
            }
            const std::uint64_t p = *p_res;
            ++primes_consumed;

            auto img_res = gcd_image_mod_p(A, B, p, gamma);
            if (!img_res) {
                return make_error<Polynomial>(img_res.error());
            }
            all_images.push_back(std::move(*img_res));
        }

        auto merge_res = merge_images(all_images, gamma, b_lm);
        if (!merge_res) {
            return make_error<Polynomial>(merge_res.error());
        }

        if (std::holds_alternative<CoprimeProven>(*merge_res)) {
            return Polynomial::constant(d);
        }

        if (const auto* cand = std::get_if<Candidate>(&*merge_res)) {
            // Trial division verification (§3.8)
            auto div_a = A.divide_exact(cand->polynomial);
            auto div_b = B.divide_exact(cand->polynomial);
            if (div_a.has_value() && div_b.has_value()) {
                // Exact candidate verified! Scale by content gcd d
                auto scaled = cand->polynomial.scale(d);
                if (!scaled) {
                    return make_error<Polynomial>(MathError::overflow);
                }
                return scaled;
            }
            // Trial division failed: all accepted primes were unlucky.
            // Discard accepted set by clearing all images and continuing.
            all_images.clear();
        }
    }

    return make_error<Polynomial>(MathError::not_converged);
}

}  // namespace nimblecas
