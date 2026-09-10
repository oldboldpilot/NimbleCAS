// Tests for nimblecas.memo_dist: content-addressed distributed memoization.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.memo_dist;
import nimblecas.testing;

using nimblecas::ContentKey;
using nimblecas::content_key;
using nimblecas::DistributedMemo;
using nimblecas::FileMemo;
using nimblecas::InProcessMemo;
using nimblecas::MathError;
using nimblecas::is_memoizable_status;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::to_hex;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A fresh directory per test, so a failure in one cannot leave state that changes another.
[[nodiscard]] auto temp_memo_dir(std::string_view tag) -> std::filesystem::path {
    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec) /
                     std::format("ncas_filememo_{}", tag);
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

auto remove_memo_dir(const std::filesystem::path& dir) -> void {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

[[nodiscard]] auto to_payload(std::string_view s) -> Payload {
    Payload p;
    p.reserve(s.size());
    for (const char c : s) {
        p.push_back(static_cast<std::byte>(c));
    }
    return p;
}

[[nodiscard]] auto as_bytes(std::string_view s) -> std::span<const std::byte> {
    return std::as_bytes(std::span(s.data(), s.size()));
}

[[nodiscard]] auto as_bytes(const Payload& p) -> std::span<const std::byte> {
    return std::span<const std::byte>(p.data(), p.size());
}

[[nodiscard]] auto make_pattern_payload(std::size_t size, std::uint8_t seed) -> Payload {
    Payload p;
    p.reserve(size);
    auto state = seed;
    for (std::size_t i = 0; i < size; ++i) {
        p.push_back(static_cast<std::byte>(state));
        state = static_cast<std::uint8_t>(state * 37u + 17u + static_cast<std::uint8_t>(i & 0xFFu));
    }
    return p;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.memo_dist")
        // -------------------------------------------------------------------
        // 1. Empty input base offsets & literal constants
        // -------------------------------------------------------------------
        .test("content_key_empty_matches_documented_offset_bases",
              [](TestContext& t) {
                  const auto key = content_key({});
                  // Spec: hi = 0xcbf29ce484222325, lo = 0x9e3779b97f4a7c15 on empty input.
                  constexpr std::uint64_t expected_hi = 0xcbf29ce484222325ULL;
                  constexpr std::uint64_t expected_lo = 0x9e3779b97f4a7c15ULL;

                  t.expect_eq(key.hi, expected_hi, "empty input hi must equal FNV-1a 64 offset basis");
                  t.expect_eq(key.lo, expected_lo, "empty input lo must equal secondary offset basis");

                  const auto hex = to_hex(key);
                  t.expect_eq(hex, std::string{"cbf29ce4842223259e3779b97f4a7c15"},
                              "to_hex(content_key({})) yields concatenated 32-char lowercase hex");
              })

        // -------------------------------------------------------------------
        // 2. Determinism, byte ordering, length dependence, 1-bit flip sensitivity
        // -------------------------------------------------------------------
        .test("content_key_determinism_and_structural_sensitivity",
              [](TestContext& t) {
                  const auto sample = as_bytes("canonical_task_encoding_v1");

                  // Determinism: multiple evaluations return identical keys
                  const auto k1 = content_key(sample);
                  const auto k2 = content_key(sample);
                  t.expect(k1 == k2, "content_key is deterministic across calls");
                  t.expect_eq(k1.hi, k2.hi, "hi matches across calls");
                  t.expect_eq(k1.lo, k2.lo, "lo matches across calls");

                  // Byte order sensitivity: permutation of bytes must change the fingerprint
                  const auto ab = content_key(as_bytes("ab"));
                  const auto ba = content_key(as_bytes("ba"));
                  t.expect(ab != ba, "content_key is sensitive to byte order ('ab' vs 'ba')");
                  t.expect(ab.hi != ba.hi || ab.lo != ba.lo, "at least one component differs on permutation");

                  const auto cat = content_key(as_bytes("cat"));
                  const auto act = content_key(as_bytes("act"));
                  t.expect(cat != act, "content_key is sensitive to byte order ('cat' vs 'act')");

                  // Length sensitivity: same prefix with different lengths or null bytes
                  const auto a = content_key(as_bytes("a"));
                  const std::array<std::byte, 2> a_null{static_cast<std::byte>('a'), static_cast<std::byte>(0)};
                  const auto a_null_key = content_key(a_null);
                  t.expect(a != a_null_key, "content_key is sensitive to length ('a' vs 'a\\0')");

                  const auto pref = content_key(as_bytes("prefix"));
                  const auto pref_ext = content_key(as_bytes("prefix_extended"));
                  t.expect(pref != pref_ext, "content_key distinguishes prefix from extended string");

                  // One-bit change sensitivity: flipping every single bit across a 16-byte block
                  auto buf = make_pattern_payload(16, 0x5A);
                  const auto base_key = content_key(as_bytes(buf));

                  bool all_bit_flips_differ = true;
                  for (std::size_t byte_idx = 0; byte_idx < buf.size(); ++byte_idx) {
                      for (unsigned bit = 0; bit < 8; ++bit) {
                          auto mutated = buf;
                          mutated[byte_idx] = static_cast<std::byte>(
                              static_cast<std::uint8_t>(mutated[byte_idx]) ^ static_cast<std::uint8_t>(1u << bit));
                          const auto mutated_key = content_key(as_bytes(mutated));
                          if (mutated_key == base_key) {
                              all_bit_flips_differ = false;
                          }
                      }
                  }
                  t.expect(all_bit_flips_differ, "every single 1-bit flip produces a distinct ContentKey");
              })

        // -------------------------------------------------------------------
        // 3. Golden Vectors (hand-derived from FNV-1a spec)
        // -------------------------------------------------------------------
        .test("content_key_golden_vectors",
              [](TestContext& t) {
                  // Golden vector 1: "a" (0x61)
                  // hi: (0xcbf29ce484222325 ^ 0x61) * 0x100000001b3 = 0xaf63dc4c8601ec8c
                  // lo: (0x9e3779b97f4a7c15 ^ (0x61 ^ 0xa5)) * 0x100000001b3 = 0x22c0a7334b921723
                  const auto k_a = content_key(as_bytes("a"));
                  t.expect_eq(k_a.hi, 0xaf63dc4c8601ec8cULL, "golden vector 'a' hi");
                  t.expect_eq(k_a.lo, 0x22c0a7334b921723ULL, "golden vector 'a' lo");
                  t.expect_eq(to_hex(k_a), std::string{"af63dc4c8601ec8c22c0a7334b921723"},
                              "golden vector 'a' hex string pinned");

                  // Golden vector 2: "hello"
                  // hi: 0xa430d84680aabd0b
                  // lo: 0x72ca750fe608607c
                  const auto k_hello = content_key(as_bytes("hello"));
                  t.expect_eq(k_hello.hi, 0xa430d84680aabd0bULL, "golden vector 'hello' hi");
                  t.expect_eq(k_hello.lo, 0x72ca750fe608607cULL, "golden vector 'hello' lo");
                  t.expect_eq(to_hex(k_hello), std::string{"a430d84680aabd0b72ca750fe608607c"},
                              "golden vector 'hello' hex string pinned");

                  // Golden vector 3: "nimblecas"
                  // hi: 0x41bd1a3d22058d55
                  // lo: 0x05f88fc36f6602ea
                  const auto k_nimblecas = content_key(as_bytes("nimblecas"));
                  t.expect_eq(k_nimblecas.hi, 0x41bd1a3d22058d55ULL, "golden vector 'nimblecas' hi");
                  t.expect_eq(k_nimblecas.lo, 0x05f88fc36f6602eaULL, "golden vector 'nimblecas' lo");
                  t.expect_eq(to_hex(k_nimblecas), std::string{"41bd1a3d22058d5505f88fc36f6602ea"},
                              "golden vector 'nimblecas' hex string pinned");

                  // Golden vector 4: "The quick brown fox jumps over the lazy dog"
                  // hi: 0xf3f9b7f5e7e47110
                  // lo: 0x856252e14ca41e7d
                  const auto k_fox = content_key(as_bytes("The quick brown fox jumps over the lazy dog"));
                  t.expect_eq(k_fox.hi, 0xf3f9b7f5e7e47110ULL, "golden vector fox hi");
                  t.expect_eq(k_fox.lo, 0x856252e14ca41e7dULL, "golden vector fox lo");
                  t.expect_eq(to_hex(k_fox), std::string{"f3f9b7f5e7e47110856252e14ca41e7d"},
                              "golden vector fox hex string pinned");

                  // Golden vector 5: binary array {0x00, 0x01, 0x02, 0x03}
                  // hi: 0x4475327f98e05411
                  // lo: 0x843e7d5de0ba0cc5
                  const std::array<std::byte, 4> bin{
                      std::byte{0x00}, std::byte{0x01}, std::byte{0x02}, std::byte{0x03}};
                  const auto k_bin = content_key(bin);
                  t.expect_eq(k_bin.hi, 0x4475327f98e05411ULL, "golden vector binary hi");
                  t.expect_eq(k_bin.lo, 0x843e7d5de0ba0cc5ULL, "golden vector binary lo");
                  t.expect_eq(to_hex(k_bin), std::string{"4475327f98e05411843e7d5de0ba0cc5"},
                              "golden vector binary hex string pinned");
              })

        // -------------------------------------------------------------------
        // 4. ContentKey ordering, formatting, and zero-padding
        // -------------------------------------------------------------------
        .test("content_key_operators_and_hex_formatting",
              [](TestContext& t) {
                  const ContentKey k_zero{0ULL, 0ULL};
                  t.expect_eq(to_hex(k_zero), std::string{"00000000000000000000000000000000"},
                              "to_hex zero pads 16 chars for both hi and lo");

                  const ContentKey k_small{1ULL, 2ULL};
                  t.expect_eq(to_hex(k_small), std::string{"00000000000000010000000000000002"},
                              "to_hex properly formats small numbers with leading zeroes");

                  const ContentKey k_max{0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
                  t.expect_eq(to_hex(k_max), std::string{"ffffffffffffffffffffffffffffffff"},
                              "to_hex handles full uint64_t max");

                  // Spaceship operator <=> consistency
                  t.expect(k_zero < k_small, "k_zero < k_small");
                  t.expect(k_small < k_max, "k_small < k_max");
                  t.expect(k_small == (ContentKey{1ULL, 2ULL}), "equality comparison holds");
                  t.expect(k_small != (ContentKey{1ULL, 3ULL}), "inequality on lo holds");
                  t.expect(k_small != (ContentKey{2ULL, 2ULL}), "inequality on hi holds");
              })

        // -------------------------------------------------------------------
        // 5. Publish then lookup hit returns exact payload
        // -------------------------------------------------------------------
        .test("publish_then_lookup_hit_returns_exact_payload",
              [](TestContext& t) {
                  InProcessMemo memo(8, 1000, std::size_t{1024} * 1024);
                  const auto key_bytes = to_payload("task_canonical_input_001");
                  const auto val_bytes = to_payload("task_output_payload_42");
                  const auto ck = content_key(as_bytes(key_bytes));

                  const auto pub_res = memo.publish(ck, as_bytes(key_bytes), as_bytes(val_bytes));
                  t.expect(pub_res.has_value(), "publish succeeds");

                  const auto lookup_res = memo.lookup(ck, as_bytes(key_bytes));
                  t.expect(lookup_res.has_value(), "lookup returns a valid Result");
                  t.expect(lookup_res->has_value(), "lookup result is engaged (hit)");
                  t.expect(*lookup_res.value() == val_bytes, "lookup returns exactly the published bytes");

                  const auto stats = memo.stats();
                  t.expect_eq(stats.publishes, std::uint64_t{1}, "1 publish recorded");
                  t.expect_eq(stats.hits, std::uint64_t{1}, "1 hit recorded");
                  t.expect_eq(stats.misses, std::uint64_t{0}, "0 misses recorded");
                  t.expect_eq(stats.key_mismatches, std::uint64_t{0}, "0 mismatches recorded");
                  t.expect_eq(stats.rejected, std::uint64_t{0}, "0 rejected recorded");
              })

        // -------------------------------------------------------------------
        // 6. Lookup of absent key is a clean miss, not an error
        // -------------------------------------------------------------------
        .test("lookup_absent_key_is_clean_miss",
              [](TestContext& t) {
                  InProcessMemo memo(8, 1000, std::size_t{1024} * 1024);
                  const auto absent_key_bytes = to_payload("key_that_was_never_published");
                  const auto ck = content_key(as_bytes(absent_key_bytes));

                  const auto res = memo.lookup(ck, as_bytes(absent_key_bytes));
                  t.expect(res.has_value(), "lookup of absent key is not an error");
                  t.expect(!res->has_value(), "lookup of absent key returns nullopt (clean miss)");

                  const auto stats = memo.stats();
                  t.expect_eq(stats.misses, std::uint64_t{1}, "stats.misses incremented");
                  t.expect_eq(stats.hits, std::uint64_t{0}, "stats.hits is zero");
                  t.expect_eq(stats.key_mismatches, std::uint64_t{0}, "stats.key_mismatches is zero");
              })

        // -------------------------------------------------------------------
        // 7. CRITICAL: Forged fingerprint collision (Exactness Rule 32)
        // -------------------------------------------------------------------
        .test("forged_fingerprint_collision_exactness_rule_32",
              [](TestContext& t) {
                  InProcessMemo memo(8, 1000, std::size_t{1024} * 1024);

                  // Two distinct full keys that we deliberately bind to the SAME ContentKey
                  const auto forged_ck = ContentKey{0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL};
                  const auto full_key_real = to_payload("computation_task_A_canonical_encoding");
                  const auto full_key_forged = to_payload("computation_task_B_DIFFERENT_PAYLOAD_HERE");
                  const auto val_real = to_payload("result_from_computation_A");

                  // Publish under forged_ck with full_key_real
                  const auto pub = memo.publish(forged_ck, as_bytes(full_key_real), as_bytes(val_real));
                  t.expect(pub.has_value(), "publish succeeds");

                  // Lookup under SAME forged_ck but DIFFERENT full_key_forged
                  const auto lookup_collision = memo.lookup(forged_ck, as_bytes(full_key_forged));
                  t.expect(lookup_collision.has_value(), "lookup returns a valid Result (not an error)");
                  t.expect(!lookup_collision->has_value(),
                           "fingerprint collision with different full key MUST return nullopt (miss)");

                  auto stats = memo.stats();
                  t.expect_eq(stats.key_mismatches, std::uint64_t{1},
                              "key_mismatches incremented on forged fingerprint collision");
                  t.expect_eq(stats.hits, std::uint64_t{0}, "no hit awarded on collision");

                  // Lookup under SAME forged_ck with real full_key STILL returns correct value
                  const auto lookup_real = memo.lookup(forged_ck, as_bytes(full_key_real));
                  t.expect(lookup_real.has_value() && lookup_real->has_value(),
                           "lookup with matching full key succeeds");
                  t.expect(*lookup_real.value() == val_real,
                           "lookup with matching full key returns exact published bytes");

                  stats = memo.stats();
                  t.expect_eq(stats.hits, std::uint64_t{1}, "hits incremented for real key");
                  t.expect_eq(stats.key_mismatches, std::uint64_t{1}, "key_mismatches unchanged");

                  // Sub-case: full key differing only in length (prefix of registered key)
                  const auto full_key_prefix = to_payload("computation_task_A");
                  const auto lookup_prefix = memo.lookup(forged_ck, as_bytes(full_key_prefix));
                  t.expect(!lookup_prefix->has_value(), "prefix full key MUST return nullopt");
                  stats = memo.stats();
                  t.expect_eq(stats.key_mismatches, std::uint64_t{2},
                              "key_mismatches incremented on length mismatch");

                  // Sub-case: full key differing only in the last single byte
                  auto full_key_near = full_key_real;
                  full_key_near.back() = static_cast<std::byte>(
                      static_cast<std::uint8_t>(full_key_near.back()) ^ 0x01u);
                  const auto lookup_near = memo.lookup(forged_ck, as_bytes(full_key_near));
                  t.expect(!lookup_near->has_value(), "single-bit difference in full key MUST return nullopt");
                  stats = memo.stats();
                  t.expect_eq(stats.key_mismatches, std::uint64_t{3},
                              "key_mismatches incremented on single-bit full key mismatch");
              })

        // -------------------------------------------------------------------
        // 8. Multiple collisions in the same bucket resolved correctly
        // -------------------------------------------------------------------
        .test("multiple_keys_in_same_bucket_coexist_correctly",
              [](TestContext& t) {
                  InProcessMemo memo(4, 1000, std::size_t{1024} * 1024);
                  // Two entries forced into the same shard (same hi) and same bucket (same lo)
                  const auto shared_ck = ContentKey{0xCAFEULL, 0xBABEULL};
                  const auto k1 = to_payload("key_1_in_shared_bucket");
                  const auto v1 = to_payload("val_1");
                  const auto k2 = to_payload("key_2_in_shared_bucket");
                  const auto v2 = to_payload("val_2");

                  t.expect(memo.publish(shared_ck, as_bytes(k1), as_bytes(v1)).has_value(), "publish k1");
                  t.expect(memo.publish(shared_ck, as_bytes(k2), as_bytes(v2)).has_value(), "publish k2");

                  const auto r1 = memo.lookup(shared_ck, as_bytes(k1));
                  const auto r2 = memo.lookup(shared_ck, as_bytes(k2));

                  t.expect(r1.has_value() && r1->has_value() && **r1 == v1, "lookup k1 returns v1");
                  t.expect(r2.has_value() && r2->has_value() && **r2 == v2, "lookup k2 returns v2");

                  // Lookup of a third absent key with the same ContentKey
                  const auto k3 = to_payload("key_3_never_published");
                  const auto r3 = memo.lookup(shared_ck, as_bytes(k3));
                  t.expect(r3.has_value() && !r3->has_value(), "lookup k3 is clean miss");
                  t.expect(memo.stats().key_mismatches > 0, "key_mismatches incremented scanning bucket");
              })

        // -------------------------------------------------------------------
        // 9. Idempotent duplicate publish (First writer wins)
        // -------------------------------------------------------------------
        .test("idempotent_duplicate_publish_first_writer_wins",
              [](TestContext& t) {
                  InProcessMemo memo(8, 1000, std::size_t{1024} * 1024);
                  const auto key = to_payload("idempotent_key");
                  const auto val1 = to_payload("initial_pure_value");
                  const auto val2 = to_payload("second_writer_value");
                  const auto ck = content_key(as_bytes(key));

                  // First publish
                  t.expect(memo.publish(ck, as_bytes(key), as_bytes(val1)).has_value(), "first publish");

                  // Second publish with same key
                  t.expect(memo.publish(ck, as_bytes(key), as_bytes(val2)).has_value(),
                           "duplicate publish succeeds (no-op)");

                  const auto stats = memo.stats();
                  t.expect_eq(stats.publishes, std::uint64_t{2}, "publishes count incremented to 2");

                  // Lookup returns the first writer's value
                  const auto res = memo.lookup(ck, as_bytes(key));
                  t.expect(res.has_value() && res->has_value(), "lookup hits");
                  t.expect(*res.value() == val1, "first writer's value is preserved");
              })

        // -------------------------------------------------------------------
        // 10. Capacity refusal: max_entries bounded no-eviction table
        // -------------------------------------------------------------------
        .test("capacity_refusal_max_entries_bounded_no_eviction",
              [](TestContext& t) {
                  constexpr std::size_t max_ent = 5;
                  InProcessMemo memo(2, max_ent, std::size_t{1024} * 1024);

                  // Fill to capacity
                  std::vector<Payload> keys;
                  std::vector<Payload> vals;
                  for (std::size_t i = 0; i < max_ent; ++i) {
                      keys.push_back(to_payload(std::format("entry_key_{:04d}", i)));
                      vals.push_back(to_payload(std::format("entry_val_{:04d}", i)));
                      const auto ck = content_key(as_bytes(keys.back()));
                      const auto pub = memo.publish(ck, as_bytes(keys.back()), as_bytes(vals.back()));
                      t.expect(pub.has_value(), "publish within capacity succeeds");
                  }

                  auto stats = memo.stats();
                  t.expect_eq(stats.publishes, static_cast<std::uint64_t>(max_ent), "max_ent publishes");
                  t.expect_eq(stats.rejected, std::uint64_t{0}, "0 rejected when within capacity");

                  // Attempt to publish 6th key past capacity
                  const auto extra_key = to_payload("entry_key_overflow_0005");
                  const auto extra_val = to_payload("entry_val_overflow_0005");
                  const auto extra_ck = content_key(as_bytes(extra_key));

                  const auto pub_extra = memo.publish(extra_ck, as_bytes(extra_key), as_bytes(extra_val));
                  // Refusing to cache is NOT an error: returns success Result<void>{}
                  t.expect(pub_extra.has_value(), "publish past capacity still returns success Result");

                  stats = memo.stats();
                  t.expect_eq(stats.rejected, std::uint64_t{1}, "stats.rejected incremented");

                  // Lookup all originally stored entries: none were evicted
                  for (std::size_t i = 0; i < max_ent; ++i) {
                      const auto ck = content_key(as_bytes(keys[i]));
                      const auto r = memo.lookup(ck, as_bytes(keys[i]));
                      t.expect(r.has_value() && r->has_value() && *r.value() == vals[i],
                               "earlier entry preserved (no eviction policy)");
                  }

                  // Lookup overflow entry: must be a clean miss
                  const auto r_extra = memo.lookup(extra_ck, as_bytes(extra_key));
                  t.expect(r_extra.has_value() && !r_extra->has_value(),
                           "unaccepted entry is a clean miss (never wrong value)");
              })

        // -------------------------------------------------------------------
        // 11. Capacity refusal: oversize value rejected
        // -------------------------------------------------------------------
        .test("capacity_refusal_oversize_value",
              [](TestContext& t) {
                  constexpr std::size_t max_val_bytes = 64;
                  InProcessMemo memo(4, 100, max_val_bytes);

                  // Value right at limit: 64 bytes
                  const auto exact_val = make_pattern_payload(64, 0x11);
                  const auto exact_key = to_payload("key_exact_64");
                  const auto ck_exact = content_key(as_bytes(exact_key));
                  t.expect(memo.publish(ck_exact, as_bytes(exact_key), as_bytes(exact_val)).has_value(),
                           "publish at limit succeeds");
                  const auto r_exact = memo.lookup(ck_exact, as_bytes(exact_key));
                  t.expect(r_exact.has_value() && r_exact->has_value() && *r_exact.value() == exact_val,
                           "lookup at limit hits");

                  // Oversize value: 65 bytes
                  const auto over_val = make_pattern_payload(65, 0x22);
                  const auto over_key = to_payload("key_over_65");
                  const auto ck_over = content_key(as_bytes(over_key));

                  const auto pub_over = memo.publish(ck_over, as_bytes(over_key), as_bytes(over_val));
                  t.expect(pub_over.has_value(), "oversize publish returns success Result");

                  const auto stats = memo.stats();
                  t.expect_eq(stats.rejected, std::uint64_t{1}, "stats.rejected incremented for oversize value");

                  const auto r_over = memo.lookup(ck_over, as_bytes(over_key));
                  t.expect(r_over.has_value() && !r_over->has_value(), "oversize value is not stored (miss)");
              })

        // -------------------------------------------------------------------
        // 12. is_memoizable_status helper
        // -------------------------------------------------------------------
        .test("is_memoizable_status_classification",
              [](TestContext& t) {
                  // 0 == ok (pure success is a deterministic function of key)
                  t.expect(is_memoizable_status(0), "status 0 (ok) is memoizable");

                  // 1 == math_error (deterministic error given identical inputs)
                  t.expect(is_memoizable_status(1), "status 1 (math_error) is memoizable");

                  // 2 == bridge_error (transport/coordination fault, must NEVER be cached)
                  t.expect(!is_memoizable_status(2), "status 2 (bridge_error) is NOT memoizable");

                  // Unknown / out-of-range statuses must all return false
                  t.expect(!is_memoizable_status(3), "status 3 is not memoizable");
                  t.expect(!is_memoizable_status(4), "status 4 is not memoizable");
                  t.expect(!is_memoizable_status(-1), "status -1 is not memoizable");
                  t.expect(!is_memoizable_status(-100), "status -100 is not memoizable");
                  t.expect(!is_memoizable_status(999), "status 999 is not memoizable");
                  t.expect(!is_memoizable_status(std::numeric_limits<int>::max()), "INT_MAX is not memoizable");
                  t.expect(!is_memoizable_status(std::numeric_limits<int>::min()), "INT_MIN is not memoizable");
              })

        // -------------------------------------------------------------------
        // 13. Shard count clamping and polymorphic interface
        // -------------------------------------------------------------------
        .test("shard_count_clamping_and_polymorphism",
              [](TestContext& t) {
                  // shard_count == 0 must be clamped to 1 without division-by-zero
                  std::unique_ptr<DistributedMemo> memo = std::make_unique<InProcessMemo>(0, 100, 1024);
                  t.expect(!memo->name().empty(), "memo name is non-empty");

                  const auto key = to_payload("poly_key");
                  const auto val = to_payload("poly_val");
                  const auto ck = content_key(as_bytes(key));

                  t.expect(memo->publish(ck, as_bytes(key), as_bytes(val)).has_value(), "publish via base ptr");
                  const auto res = memo->lookup(ck, as_bytes(key));
                  t.expect(res.has_value() && res->has_value() && *res.value() == val, "lookup via base ptr");
              })

        // -------------------------------------------------------------------
        // 14. Large payloads and empty payloads
        // -------------------------------------------------------------------
        .test("large_and_empty_payloads",
              [](TestContext& t) {
                  InProcessMemo memo(8, 100, std::size_t{4} * 1024 * 1024);

                  // Empty key and empty value
                  const auto empty_k = Payload{};
                  const auto empty_v = Payload{};
                  const auto ck_empty = content_key(as_bytes(empty_k));

                  t.expect(memo.publish(ck_empty, as_bytes(empty_k), as_bytes(empty_v)).has_value(),
                           "publish empty key/value");
                  const auto r_empty = memo.lookup(ck_empty, as_bytes(empty_k));
                  t.expect(r_empty.has_value() && r_empty->has_value() && r_empty->value().empty(),
                           "lookup of empty key returns empty value");

                  // Large 512KB payload
                  const auto large_val = make_pattern_payload(std::size_t{512} * 1024, 0x7E);
                  const auto large_key = to_payload("large_payload_key");
                  const auto ck_large = content_key(as_bytes(large_key));

                  t.expect(memo.publish(ck_large, as_bytes(large_key), as_bytes(large_val)).has_value(),
                           "publish 512KB value");
                  const auto r_large = memo.lookup(ck_large, as_bytes(large_key));
                  t.expect(r_large.has_value() && r_large->has_value() && *r_large.value() == large_val,
                           "512KB payload recovered bit-identically");
              })

        // -------------------------------------------------------------------
        // 15. Concurrency: highly concurrent publish and lookup across threads
        // -------------------------------------------------------------------
        .test("concurrent_publish_and_lookup_race_free_and_consistent",
              [](TestContext& t) {
                  constexpr std::size_t shard_count = 16;
                  constexpr std::size_t total_keys = 200;
                  constexpr int num_threads = 16;
                  constexpr int ops_per_thread = 200;

                  InProcessMemo memo(shard_count, 10'000, std::size_t{1024} * 1024);

                  std::vector<Payload> keys(total_keys);
                  std::vector<Payload> vals(total_keys);
                  std::vector<ContentKey> cks(total_keys);

                  for (std::size_t i = 0; i < total_keys; ++i) {
                      keys[i] = to_payload(std::format("concurrent_task_key_{:05d}", i));
                      vals[i] = to_payload(std::format("concurrent_task_val_{:05d}_payload_bytes", i));
                      cks[i] = content_key(as_bytes(keys[i]));
                  }

                  // Which keys a writer actually published. Not every key is: the index is
                  // (tid*17 + op*31) % total_keys and only every third op writes. Without this
                  // the "was anything LOST?" half of the verification below cannot be checked,
                  // and a key silently dropped by a race would pass unnoticed.
                  std::vector<std::atomic<bool>> published(total_keys);
                  for (auto& flag : published) {
                      flag.store(false, std::memory_order_relaxed);
                  }

                  std::atomic<bool> start_gate{false};
                  std::atomic<std::size_t> wrong_values{0};
                  std::atomic<std::size_t> lookup_hits{0};
                  std::atomic<std::size_t> lookup_misses{0};

                  std::vector<std::thread> threads;
                  threads.reserve(num_threads);

                  for (int tid = 0; tid < num_threads; ++tid) {
                      threads.emplace_back([&, tid] {
                          while (!start_gate.load(std::memory_order_acquire)) {
                              std::this_thread::yield();
                          }

                          for (int op = 0; op < ops_per_thread; ++op) {
                              const std::size_t key_idx = static_cast<std::size_t>((tid * 17 + op * 31) % total_keys);

                              if (op % 3 == 0) {
                                  // Publish assigned key
                                  const auto res = memo.publish(cks[key_idx], as_bytes(keys[key_idx]),
                                                                as_bytes(vals[key_idx]));
                                  if (!res.has_value()) {
                                      wrong_values.fetch_add(1, std::memory_order_relaxed);
                                  } else {
                                      published[key_idx].store(true, std::memory_order_relaxed);
                                  }
                              } else {
                                  // Lookup key
                                  const auto res = memo.lookup(cks[key_idx], as_bytes(keys[key_idx]));
                                  if (!res.has_value()) {
                                      wrong_values.fetch_add(1, std::memory_order_relaxed);
                                  } else if (res->has_value()) {
                                      lookup_hits.fetch_add(1, std::memory_order_relaxed);
                                      if (*res.value() != vals[key_idx]) {
                                          wrong_values.fetch_add(1, std::memory_order_relaxed);
                                      }
                                  } else {
                                      lookup_misses.fetch_add(1, std::memory_order_relaxed);
                                  }
                              }
                          }
                      });
                  }

                  start_gate.store(true, std::memory_order_release);
                  for (auto& th : threads) {
                      th.join();
                  }

                  t.expect_eq(wrong_values.load(), std::size_t{0},
                              "zero corrupted or wrong values observed during concurrent ops");

                  // Post-concurrency verification: ensure every published key is fully intact
                  std::size_t post_missing = 0;
                  std::size_t post_corrupt = 0;
                  for (std::size_t i = 0; i < total_keys; ++i) {
                      // Attempt lookup
                      const auto res = memo.lookup(cks[i], as_bytes(keys[i]));
                      if (res.has_value() && res->has_value()) {
                          if (*res.value() != vals[i]) {
                              ++post_corrupt;
                          }
                      } else if (published[i].load(std::memory_order_relaxed)) {
                          // Published during the race, absent afterwards: the table lost it.
                          ++post_missing;
                      }
                  }

                  t.expect_eq(post_corrupt, std::size_t{0},
                              "post-concurrency verification: no corrupted values in table");
                  t.expect_eq(post_missing, std::size_t{0},
                              "post-concurrency verification: no published key went missing");

                  const auto final_stats = memo.stats();
                  t.expect_eq(final_stats.key_mismatches, std::uint64_t{0},
                              "zero key mismatches during race-free concurrent execution");
                  t.expect_eq(final_stats.rejected, std::uint64_t{0},
                              "zero rejected publishes during concurrent execution");
              })

        // ── FileMemo: the same table, made durable ────────────────────────────────────
        .test("file_memo_survives_process_restart",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("restart");
                  const auto path = dir / "memo.log";

                  const auto k1 = to_payload("durable_task_key_one");
                  const auto v1 = to_payload("durable_task_value_one");
                  const auto k2 = to_payload("durable_task_key_two");
                  const auto v2 = to_payload("durable_task_value_two");

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "FileMemo::create succeeds on a fresh path");
                      if (!memo.has_value()) { return; }
                      t.expect((*memo)->publish(content_key(as_bytes(k1)), as_bytes(k1), as_bytes(v1)).has_value(), "publish k1");
                      t.expect((*memo)->publish(content_key(as_bytes(k2)), as_bytes(k2), as_bytes(v2)).has_value(), "publish k2");
                      t.expect_eq((*memo)->size(), std::size_t{2}, "two records indexed before close");
                  }  // destroyed: stands in for the process exiting

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "FileMemo::create reopens the existing log");
                      if (!memo.has_value()) { return; }
                      t.expect_eq((*memo)->size(), std::size_t{2}, "both records recovered from disk");
                      t.expect(!(*memo)->log_was_damaged(), "no damage reported on a clean log");

                      const auto r1 = (*memo)->lookup(content_key(as_bytes(k1)), as_bytes(k1));
                      t.expect(r1.has_value() && r1->has_value(), "k1 hits after restart");
                      if (r1.has_value() && r1->has_value()) {
                          t.expect(**r1 == v1, "k1 returns exactly the bytes published before restart");
                      }
                      const auto r2 = (*memo)->lookup(content_key(as_bytes(k2)), as_bytes(k2));
                      t.expect(r2.has_value() && r2->has_value() && **r2 == v2,
                               "k2 returns exactly the bytes published before restart");

                      const auto absent = to_payload("never_published");
                      const auto r3 = (*memo)->lookup(content_key(as_bytes(absent)), as_bytes(absent));
                      t.expect(r3.has_value() && !r3->has_value(), "an absent key is a clean miss after restart");
                  }
                  remove_memo_dir(dir);
              })

        .test("file_memo_torn_tail_is_a_miss_not_a_wrong_value",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("torn");
                  const auto path = dir / "memo.log";

                  const auto k1 = to_payload("intact_record_key");
                  const auto v1 = to_payload("intact_record_value");
                  const auto k2 = to_payload("torn_record_key");
                  const auto v2 = to_payload("torn_record_value_that_is_long_enough_to_cut");

                  std::uintmax_t good_prefix = 0;
                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { return; }
                      (void)(*memo)->publish(content_key(as_bytes(k1)), as_bytes(k1), as_bytes(v1));
                      good_prefix = (*memo)->good_prefix_bytes();
                      (void)(*memo)->publish(content_key(as_bytes(k2)), as_bytes(k2), as_bytes(v2));
                  }

                  // Simulate a crash mid-append: cut the file so the SECOND record is partial.
                  std::error_code fec;
                  const auto full_size = std::filesystem::file_size(path, fec);
                  t.expect(!fec && full_size > good_prefix, "the second record actually occupies bytes");
                  std::filesystem::resize_file(path, good_prefix + (full_size - good_prefix) / 2, fec);
                  t.expect(!fec, "the log can be truncated to simulate a crash");

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "a torn log still opens");
                      if (!memo.has_value()) { return; }

                      t.expect_eq((*memo)->size(), std::size_t{1}, "only the intact record is indexed");
                      t.expect((*memo)->log_was_damaged(), "the torn tail is reported, not hidden");
                      t.expect_eq((*memo)->good_prefix_bytes(), good_prefix,
                                  "the good prefix ends exactly at the last complete record");

                      const auto r1 = (*memo)->lookup(content_key(as_bytes(k1)), as_bytes(k1));
                      t.expect(r1.has_value() && r1->has_value() && **r1 == v1,
                               "the record before the tear is still exact");

                      // THE POINT: the torn record must be a MISS, never a partial value.
                      const auto r2 = (*memo)->lookup(content_key(as_bytes(k2)), as_bytes(k2));
                      t.expect(r2.has_value(), "a torn record does not make lookup an error");
                      t.expect(!r2->has_value(), "a torn record is a MISS, never a truncated value");

                      // And the store must still be writable: a new publish overwrites the
                      // damaged tail rather than appending past it.
                      const auto k3 = to_payload("after_the_tear_key");
                      const auto v3 = to_payload("after_the_tear_value");
                      t.expect((*memo)->publish(content_key(as_bytes(k3)), as_bytes(k3), as_bytes(v3)).has_value(),
                               "publishing after a tear succeeds");
                      const auto r3 = (*memo)->lookup(content_key(as_bytes(k3)), as_bytes(k3));
                      t.expect(r3.has_value() && r3->has_value() && **r3 == v3, "the new record reads back");
                  }

                  // ...and it survives one more restart, proving the repair was durable.
                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "reopen after repair");
                      if (!memo.has_value()) { return; }
                      t.expect_eq((*memo)->size(), std::size_t{2}, "intact record plus the one written over the tear");
                      t.expect(!(*memo)->log_was_damaged(),
                               "the tear was cut at open, so the repaired log reopens CLEAN");
                      const auto k3 = to_payload("after_the_tear_key");
                      const auto r3 = (*memo)->lookup(content_key(as_bytes(k3)), as_bytes(k3));
                      t.expect(r3.has_value() && r3->has_value(), "the repaired record persists");
                  }
                  remove_memo_dir(dir);
              })

        .test("file_memo_corrupt_record_is_a_miss",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("corrupt");
                  const auto path = dir / "memo.log";
                  const auto k = to_payload("corruptible_key");
                  const auto v = to_payload("corruptible_value_bytes");

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { return; }
                      (void)(*memo)->publish(content_key(as_bytes(k)), as_bytes(k), as_bytes(v));
                  }

                  // Flip one bit INSIDE the stored value, leaving the framing intact. Only the
                  // checksum can catch this; without it the store would serve altered bytes.
                  std::error_code sec;
                  const auto size = std::filesystem::file_size(path, sec);
                  t.expect(!sec && size > 20, "record written");
                  {
                      std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
                      t.expect(static_cast<bool>(f), "reopen the log for corruption");
                      const auto pos = static_cast<std::streamoff>(size - 12);
                      f.seekg(pos, std::ios::beg);
                      char byte = 0;
                      f.read(&byte, 1);
                      byte = static_cast<char>(static_cast<unsigned char>(byte) ^ 0x40U);
                      f.seekp(pos, std::ios::beg);
                      f.write(&byte, 1);
                  }

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "a corrupt log still opens");
                      if (!memo.has_value()) { return; }
                      const auto r = (*memo)->lookup(content_key(as_bytes(k)), as_bytes(k));
                      t.expect(r.has_value(), "corruption does not make lookup an error");
                      t.expect(!r->has_value(), "a checksum failure is a MISS, never the altered bytes");
                  }
                  remove_memo_dir(dir);
              })

        .test("file_memo_forged_collision_never_serves_the_wrong_answer",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("collide");
                  const auto path = dir / "memo.log";
                  auto memo = FileMemo::create(path);
                  t.expect(memo.has_value(), "create");
                  if (!memo.has_value()) { return; }

                  // Two DIFFERENT full keys deliberately published under one ContentKey.
                  const auto forged = ContentKey{0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
                  const auto key_a = to_payload("file_memo_task_A");
                  const auto val_a = to_payload("result_of_A");
                  const auto key_b = to_payload("file_memo_task_B_different");

                  t.expect((*memo)->publish(forged, as_bytes(key_a), as_bytes(val_a)).has_value(), "publish under the forged key");

                  const auto hit = (*memo)->lookup(forged, as_bytes(key_a));
                  t.expect(hit.has_value() && hit->has_value() && **hit == val_a,
                           "the matching full key returns exactly its own value");

                  const auto collide = (*memo)->lookup(forged, as_bytes(key_b));
                  t.expect(collide.has_value(), "a collision is not an error");
                  t.expect(!collide->has_value(),
                           "a fingerprint collision with a different full key MUST miss, never serve A's value");
                  t.expect_eq((*memo)->stats().key_mismatches, std::uint64_t{1},
                              "the collision is counted as a genuine key mismatch");
                  remove_memo_dir(dir);
              })

        .test("file_memo_is_interchangeable_with_in_process_memo",
              [](TestContext& t) {
                  // Same operations, same observable answers: FileMemo must be a drop-in
                  // DistributedMemo, since the executor holds only the base interface.
                  const auto dir = temp_memo_dir("iface");
                  const auto path = dir / "memo.log";
                  auto file_memo = FileMemo::create(path);
                  t.expect(file_memo.has_value(), "create");
                  if (!file_memo.has_value()) { return; }

                  InProcessMemo mem;
                  DistributedMemo& a = mem;
                  DistributedMemo& b = **file_memo;

                  t.expect(a.name() == "in_process", "in-process reports its name");
                  t.expect(b.name() == "file", "file-backed reports its name");

                  bool all_agree = true;
                  for (int i = 0; i < 24; ++i) {
                      const auto key = to_payload(std::format("iface_key_{:03d}", i));
                      const auto val = to_payload(std::format("iface_val_{:03d}_payload", i));
                      const auto ck = content_key(as_bytes(key));

                      const auto miss_a = a.lookup(ck, as_bytes(key));
                      const auto miss_b = b.lookup(ck, as_bytes(key));
                      all_agree = all_agree && miss_a.has_value() && miss_b.has_value() &&
                                  !miss_a->has_value() && !miss_b->has_value();

                      (void)a.publish(ck, as_bytes(key), as_bytes(val));
                      (void)b.publish(ck, as_bytes(key), as_bytes(val));

                      const auto hit_a = a.lookup(ck, as_bytes(key));
                      const auto hit_b = b.lookup(ck, as_bytes(key));
                      all_agree = all_agree && hit_a.has_value() && hit_b.has_value() &&
                                  hit_a->has_value() && hit_b->has_value() && **hit_a == val &&
                                  **hit_b == val;
                  }
                  t.expect(all_agree, "both implementations answer identically for 24 keys");
                  t.expect_eq(a.stats().hits, b.stats().hits, "hit counts agree");
                  t.expect_eq(a.stats().misses, b.stats().misses, "miss counts agree");
                  t.expect_eq(a.stats().publishes, b.stats().publishes, "publish counts agree");
                  remove_memo_dir(dir);
              })

        .test("file_memo_duplicate_publish_keeps_the_first_and_does_not_grow",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("dup");
                  const auto path = dir / "memo.log";
                  auto memo = FileMemo::create(path);
                  t.expect(memo.has_value(), "create");
                  if (!memo.has_value()) { return; }

                  const auto k = to_payload("repeatedly_published_key");
                  const auto v = to_payload("the_one_true_value");
                  const auto ck = content_key(as_bytes(k));

                  (void)(*memo)->publish(ck, as_bytes(k), as_bytes(v));
                  const auto after_first = (*memo)->good_prefix_bytes();
                  for (int i = 0; i < 5; ++i) {
                      (void)(*memo)->publish(ck, as_bytes(k), as_bytes(v));
                  }
                  t.expect_eq((*memo)->good_prefix_bytes(), after_first,
                              "re-publishing an existing key appends nothing");
                  t.expect_eq((*memo)->size(), std::size_t{1}, "still exactly one record");
                  t.expect_eq((*memo)->stats().publishes, std::uint64_t{6},
                              "every publish call still counts as a publish");
                  const auto r = (*memo)->lookup(ck, as_bytes(k));
                  t.expect(r.has_value() && r->has_value() && **r == v, "the value is unchanged");
                  remove_memo_dir(dir);
              })

        .test("file_memo_oversize_value_is_refused_not_an_error",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("oversize");
                  const auto path = dir / "memo.log";
                  auto memo = FileMemo::create(path, /*max_value_bytes=*/64);
                  t.expect(memo.has_value(), "create with a tiny value bound");
                  if (!memo.has_value()) { return; }

                  const auto k = to_payload("oversize_key");
                  const Payload big(256, std::byte{0x5A});
                  const auto ck = content_key(as_bytes(k));

                  const auto pub = (*memo)->publish(ck, as_bytes(k), std::span<const std::byte>(big));
                  t.expect(pub.has_value(), "refusing to cache is SUCCESS, not an error");
                  t.expect_eq((*memo)->stats().rejected, std::uint64_t{1}, "the refusal is counted");
                  t.expect_eq((*memo)->good_prefix_bytes(), std::uintmax_t{0}, "nothing was written");
                  const auto r = (*memo)->lookup(ck, as_bytes(k));
                  t.expect(r.has_value() && !r->has_value(), "and it reads back as a plain miss");
                  remove_memo_dir(dir);
              })

        .test("file_memo_bad_path_is_an_honest_error",
              [](TestContext& t) {
                  // A directory that does not exist is a runtime condition, not a crash.
                  std::error_code ec;
                  const std::filesystem::path bad =
                      std::filesystem::temp_directory_path(ec) / "ncas_no_such_dir_m9" / "sub" / "memo.log";
                  std::filesystem::remove_all(bad.parent_path().parent_path(), ec);
                  const auto memo = FileMemo::create(bad);
                  t.expect(!memo.has_value(), "creating under a missing directory fails");
                  if (!memo.has_value()) {
                      t.expect(memo.error() == MathError::domain_error,
                               "and fails with an honest domain_error, not a crash");
                  }
              })

        // A good record BEHIND a bad one must NOT be loaded. This is the claim the module
        // emphasises most and, until this test, the one nothing exercised: every earlier
        // corruption case damaged the last record, where "stop at the first bad one" and
        // "stop at the end" are indistinguishable.
        .test("file_memo_stops_at_the_first_bad_record_and_does_not_scan_past_it",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("behind");
                  const auto path = dir / "memo.log";

                  const auto k1 = to_payload("record_one_key");
                  const auto v1 = to_payload("record_one_value");
                  const auto k2 = to_payload("record_two_key");
                  const auto v2 = to_payload("record_two_value");
                  const auto k3 = to_payload("record_three_key");
                  const auto v3 = to_payload("record_three_value");

                  std::uintmax_t after_first = 0;
                  std::uintmax_t after_second = 0;
                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      (void)(*memo)->publish(content_key(as_bytes(k1)), as_bytes(k1), as_bytes(v1));
                      after_first = (*memo)->good_prefix_bytes();
                      (void)(*memo)->publish(content_key(as_bytes(k2)), as_bytes(k2), as_bytes(v2));
                      after_second = (*memo)->good_prefix_bytes();
                      (void)(*memo)->publish(content_key(as_bytes(k3)), as_bytes(k3), as_bytes(v3));
                      t.expect_eq((*memo)->size(), std::size_t{3}, "three records written");
                  }

                  // Damage a byte INSIDE record two, leaving records one and three untouched
                  // and all framing intact, so only the checksum can catch it.
                  {
                      std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
                      t.expect(static_cast<bool>(f), "reopen for corruption");
                      const auto pos = static_cast<std::streamoff>(after_second - 4);
                      f.seekg(pos, std::ios::beg);
                      char byte = 0;
                      f.read(&byte, 1);
                      byte = static_cast<char>(static_cast<unsigned char>(byte) ^ 0x20U);
                      f.seekp(pos, std::ios::beg);
                      f.write(&byte, 1);
                  }

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "a log damaged in the middle still opens");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }

                      t.expect_eq((*memo)->size(), std::size_t{1},
                                  "ONLY the record before the damage is indexed");
                      t.expect((*memo)->log_was_damaged(), "the damage is reported");
                      t.expect_eq((*memo)->good_prefix_bytes(), after_first,
                                  "the good prefix ends at the last record before the damage");

                      const auto r1 = (*memo)->lookup(content_key(as_bytes(k1)), as_bytes(k1));
                      t.expect(r1.has_value() && r1->has_value() && **r1 == v1,
                               "the record before the damage is still exact");

                      const auto r2 = (*memo)->lookup(content_key(as_bytes(k2)), as_bytes(k2));
                      t.expect(r2.has_value() && !r2->has_value(), "the damaged record misses");

                      // THE POINT: record three was perfectly intact, and must STILL be gone.
                      // Serving it would mean the loader scanned past a record it could not
                      // verify, which is how a corrupt log starts yielding plausible garbage.
                      const auto r3 = (*memo)->lookup(content_key(as_bytes(k3)), as_bytes(k3));
                      t.expect(r3.has_value() && !r3->has_value(),
                               "an INTACT record behind the damaged one is NOT served");
                  }
                  remove_memo_dir(dir);
              })

        // A repair record SHORTER than the damaged remnant. Without truncation at open the
        // remnant's tail would survive past the repair, leaving the store permanently flagged
        // as damaged and putting former value bytes on a record boundary.
        .test("file_memo_repair_shorter_than_the_damage_still_leaves_a_clean_log",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("shortfix");
                  const auto path = dir / "memo.log";

                  const auto k1 = to_payload("kept_key");
                  const auto v1 = to_payload("kept_value");
                  const auto k2 = to_payload("long_record_key_padded_out_considerably");
                  const Payload v2(400, std::byte{0x7E});

                  std::uintmax_t after_first = 0;
                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      (void)(*memo)->publish(content_key(as_bytes(k1)), as_bytes(k1), as_bytes(v1));
                      after_first = (*memo)->good_prefix_bytes();
                      (void)(*memo)->publish(content_key(as_bytes(k2)), as_bytes(k2),
                                             std::span<const std::byte>(v2));
                  }

                  // Cut most of the long record away, leaving a substantial remnant.
                  std::error_code rec;
                  const auto full = std::filesystem::file_size(path, rec);
                  t.expect(!rec && full > after_first + 200, "the long record is long");
                  std::filesystem::resize_file(path, after_first + 200, rec);
                  t.expect(!rec, "truncate to leave a 200-byte remnant");

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "opens");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      t.expect((*memo)->log_was_damaged(), "the remnant is reported");
                      std::error_code cec;
                      t.expect_eq(std::filesystem::file_size(path, cec), after_first,
                                  "the remnant is CUT at open, not left in place");

                      // A repair record far shorter than the 200-byte remnant that was there.
                      const auto k3 = to_payload("s");
                      const auto v3 = to_payload("t");
                      t.expect((*memo)->publish(content_key(as_bytes(k3)), as_bytes(k3), as_bytes(v3)).has_value(),
                               "the short repair publishes");
                  }

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "reopens after the short repair");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      t.expect(!(*memo)->log_was_damaged(),
                               "the log is CLEAN afterwards -- no permanently standing damage flag");
                      t.expect_eq((*memo)->size(), std::size_t{2}, "the kept record and the repair");
                  }
                  remove_memo_dir(dir);
              })

        // publish() indexes under the ContentKey the CALLER supplied, and the interface nowhere
        // requires that to equal content_key(full_key). The record therefore STORES the key
        // rather than recomputing it on load; otherwise such an entry would be re-keyed on
        // restart and become unreachable, so FileMemo would honour a different contract than
        // InProcessMemo across exactly one event.
        .test("file_memo_keeps_a_caller_chosen_content_key_across_restart",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("rekey");
                  const auto path = dir / "memo.log";

                  const auto forged = ContentKey{0xABCDEF0123456789ULL, 0x1122334455667788ULL};
                  const auto key = to_payload("bytes_whose_content_key_is_something_else");
                  const auto val = to_payload("value_under_a_caller_chosen_key");
                  t.expect(!(content_key(as_bytes(key)) == forged),
                           "the chosen key really differs from content_key(full_key)");

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      t.expect((*memo)->publish(forged, as_bytes(key), as_bytes(val)).has_value(), "publish");
                      const auto hit = (*memo)->lookup(forged, as_bytes(key));
                      t.expect(hit.has_value() && hit->has_value() && **hit == val,
                               "it hits before restart");
                  }

                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "reopen");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      const auto hit = (*memo)->lookup(forged, as_bytes(key));
                      t.expect(hit.has_value() && hit->has_value(),
                               "and it STILL hits after restart -- the key was stored, not recomputed");
                      if (hit.has_value() && hit->has_value()) {
                          t.expect(**hit == val, "with exactly the published bytes");
                      }
                  }
                  remove_memo_dir(dir);
              })

        // The published format spec must describe the actual bytes. Written little-endian, the
        // magic constant has to put 'N','C','M','R' on disk in that order -- an independent
        // reader built from the documentation must be able to find them.
        .test("file_memo_magic_bytes_on_disk_match_the_documented_spec",
              [](TestContext& t) {
                  const auto dir = temp_memo_dir("magic");
                  const auto path = dir / "memo.log";
                  {
                      auto memo = FileMemo::create(path);
                      t.expect(memo.has_value(), "create");
                      if (!memo.has_value()) { remove_memo_dir(dir); return; }
                      const auto k = to_payload("magic_key");
                      const auto v = to_payload("magic_value");
                      (void)(*memo)->publish(content_key(as_bytes(k)), as_bytes(k), as_bytes(v));
                  }
                  std::ifstream f(path, std::ios::binary);
                  t.expect(static_cast<bool>(f), "reopen the log to inspect its bytes");
                  std::array<char, 4> magic{};
                  f.read(magic.data(), 4);
                  t.expect_eq(f.gcount(), std::streamsize{4}, "four magic bytes present");
                  t.expect(magic[0] == 'N' && magic[1] == 'C' && magic[2] == 'M' && magic[3] == 'R',
                           "the file literally begins with the documented magic NCMR");
                  remove_memo_dir(dir);
              })
        .run();
}
