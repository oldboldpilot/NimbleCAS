// Tests for nimblecas.rng: determinism, the parallel partition-independence invariant,
// stream splitting, and distribution bounds.
// @author Olumuyiwa Oluwasanmi
//
// The central test is the parallelism invariant: because the counter core is stateless,
// concatenating the outputs of several disjoint counter sub-ranges (as independent
// workers would) must reproduce the single-threaded sequence exactly.

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.testing;

using nimblecas::Rng;
using nimblecas::counter_u64;
using nimblecas::counter_u64_batch;
using nimblecas::splitmix64;
using nimblecas::uniform_double;
using nimblecas::uniform_int;
using nimblecas::uniform_unit;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    return TestSuite("nimblecas.rng")
        .test("seeded_is_deterministic",
              [](TestContext& t) {
                  // Two generators from the same seed produce identical first-K draws.
                  auto a = Rng::seeded(42);
                  auto b = Rng::seeded(42);
                  bool identical = true;
                  for (int i = 0; i < 64; ++i) {
                      if (a.next_u64() != b.next_u64()) {
                          identical = false;
                      }
                  }
                  t.expect(identical, "Rng::seeded(42) reproduces its sequence exactly");

                  // A different seed diverges (overwhelmingly likely on the first draw).
                  auto c = Rng::seeded(43);
                  auto d = Rng::seeded(42);
                  t.expect_ne(c.next_u64(), d.next_u64(),
                              "different seeds produce different first draws");
              })
        .test("parallelism_partition_independence",
              [](TestContext& t) {
                  // THE core invariant. Generate counter_u64(key, 0..N-1) sequentially,
                  // then reproduce the same indices in several disjoint ranges (as
                  // parallel workers would) and concatenate; the results must match
                  // element-wise, proving the stream is independent of partitioning.
                  constexpr std::uint64_t key = 0xABCDEF0123456789ULL;
                  constexpr std::uint64_t n = 1000;

                  std::vector<std::uint64_t> sequential;
                  sequential.reserve(n);
                  for (std::uint64_t i = 0; i < n; ++i) {
                      sequential.push_back(counter_u64(key, i));
                  }

                  // Partition [0, n) into ragged, disjoint slices covering it exactly.
                  const std::array<std::pair<std::uint64_t, std::uint64_t>, 4> slices{{
                      {0, 137}, {137, 500}, {500, 501}, {501, n},
                  }};
                  std::vector<std::uint64_t> partitioned;
                  partitioned.reserve(n);
                  for (const auto& [begin, end] : slices) {
                      for (std::uint64_t i = begin; i < end; ++i) {
                          partitioned.push_back(counter_u64(key, i));
                      }
                  }

                  t.expect_eq(partitioned.size(), sequential.size(),
                              "partitioned run covers the whole index range");
                  bool elementwise_equal = partitioned.size() == sequential.size();
                  for (std::size_t i = 0; i < sequential.size() && i < partitioned.size(); ++i) {
                      if (sequential[i] != partitioned[i]) {
                          elementwise_equal = false;
                      }
                  }
                  t.expect(elementwise_equal,
                           "partitioned stream is element-wise identical to sequential run");

                  // Adjacent counters must not be trivially correlated / equal.
                  t.expect_ne(counter_u64(key, 0), counter_u64(key, 1),
                              "adjacent counters decorrelate");
              })
        .test("split_streams_are_independent",
              [](TestContext& t) {
                  auto parent = Rng::seeded(7);
                  auto child0 = parent.split(0);
                  auto child1 = parent.split(1);
                  auto child2 = parent.split(2);

                  // splitting must not disturb the parent.
                  t.expect_eq(parent.counter(), std::uint64_t{0},
                              "split does not advance the parent");

                  const std::uint64_t p = Rng::seeded(7).next_u64();
                  const std::uint64_t v0 = child0.next_u64();
                  const std::uint64_t v1 = child1.next_u64();
                  const std::uint64_t v2 = child2.next_u64();

                  t.expect_ne(v0, v1, "split(0) and split(1) differ");
                  t.expect_ne(v1, v2, "split(1) and split(2) differ");
                  t.expect_ne(v0, v2, "split(0) and split(2) differ");
                  t.expect_ne(v0, p, "split(0) differs from the parent's own draw");
                  t.expect_ne(v1, p, "split(1) differs from the parent's own draw");

                  // Splitting is itself deterministic.
                  t.expect_eq(Rng::seeded(7).split(0).next_u64(), v0,
                              "split(0) is reproducible");
              })
        .test("uniform_unit_in_range",
              [](TestContext& t) {
                  auto rng = Rng::seeded(123);
                  bool in_range = true;
                  for (int i = 0; i < 100000; ++i) {
                      const double u = rng.next_unit();
                      if (!(u >= 0.0 && u < 1.0)) {
                          in_range = false;
                      }
                  }
                  t.expect(in_range, "next_unit() always lies in [0, 1)");

                  // Boundary draws of the free function.
                  t.expect(uniform_unit(0) == 0.0, "uniform_unit(0) == 0.0");
                  const double top = uniform_unit(std::numeric_limits<std::uint64_t>::max());
                  t.expect(top < 1.0, "uniform_unit(all-ones) is strictly below 1.0");
              })
        .test("uniform_int_within_inclusive_range",
              [](TestContext& t) {
                  auto rng = Rng::seeded(555);
                  bool in_range = true;
                  bool saw_lo = false;
                  bool saw_hi = false;
                  constexpr std::int64_t lo = -5;
                  constexpr std::int64_t hi = 9;
                  for (int i = 0; i < 100000; ++i) {
                      auto r = rng.next_int(lo, hi);
                      if (!r) {
                          in_range = false;
                          break;
                      }
                      if (*r < lo || *r > hi) {
                          in_range = false;
                      }
                      if (*r == lo) saw_lo = true;
                      if (*r == hi) saw_hi = true;
                  }
                  t.expect(in_range, "next_int(-5, 9) always within [-5, 9]");
                  t.expect(saw_lo && saw_hi, "both endpoints of the inclusive range are reachable");

                  // Degenerate single-point range.
                  auto single = uniform_int(0xDEADBEEF, 3, 3);
                  t.expect(single.has_value() && *single == 3,
                           "uniform_int(_, 3, 3) is always 3");
              })
        .test("distribution_domain_errors",
              [](TestContext& t) {
                  auto bad_d = uniform_double(0, 5.0, 1.0);
                  t.expect(!bad_d.has_value(), "uniform_double(hi<lo) fails");
                  t.expect(bad_d.error() == nimblecas::MathError::domain_error,
                           "uniform_double(hi<lo) yields domain_error");

                  auto bad_i = uniform_int(0, 10, 2);
                  t.expect(!bad_i.has_value(), "uniform_int(hi<lo) fails");
                  t.expect(bad_i.error() == nimblecas::MathError::domain_error,
                           "uniform_int(hi<lo) yields domain_error");

                  // Valid uniform_double stays in [lo, hi).
                  auto ok_d = uniform_double(0x1234567890ABCDEFULL, -2.0, 2.0);
                  t.expect(ok_d.has_value(), "uniform_double(valid) succeeds");
                  if (ok_d) {
                      t.expect(*ok_d >= -2.0 && *ok_d < 2.0,
                               "uniform_double result lies in [lo, hi)");
                  }

                  // A width that overflows to +inf must fail on the railway, not return a
                  // silent NaN (bits==0 would otherwise give 0 * inf = NaN with has_value()).
                  auto wide = uniform_double(0, -1e308, 1e308);
                  t.expect(!wide.has_value(), "uniform_double(over-wide range) fails");
                  t.expect(!wide.has_value() && wide.error() == nimblecas::MathError::overflow,
                           "over-wide uniform_double yields overflow, not a NaN success");
              })
        .test("uniform_unit_mean_is_smoke_ok",
              [](TestContext& t) {
                  // Loose smoke test: the mean of many draws should sit near 0.5.
                  auto rng = Rng::seeded(2024);
                  constexpr int draws = 100000;
                  double sum = 0.0;
                  for (int i = 0; i < draws; ++i) {
                      sum += rng.next_unit();
                  }
                  const double mean = sum / static_cast<double>(draws);
                  t.expect(std::abs(mean - 0.5) < 0.05,
                           std::format("mean of {} uniform_unit draws ~ 0.5 (got {})",
                                       draws, mean));
              })
        .test("splitmix64_is_pure_and_varies",
              [](TestContext& t) {
                  // Sanity on the mixing primitive: deterministic and non-trivial.
                  t.expect_eq(splitmix64(0), splitmix64(0), "splitmix64 is deterministic");
                  t.expect_ne(splitmix64(0), splitmix64(1),
                              "splitmix64 separates adjacent inputs");
              })
        .test("counter_u64_batch_is_bit_identical_to_scalar",
              [](TestContext& t) {
                  // The AVX-512 (or scalar-fallback) batch core must reproduce the scalar
                  // counter_u64 loop bit-for-bit — this is the reproducibility contract the
                  // whole parallel design rests on. Cover sizes that span several full 8-wide
                  // vector groups plus every possible tail remainder (n % 8), across a base
                  // counter chosen so base + n crosses no special boundary.
                  constexpr std::uint64_t key = 0xDEADBEEFCAFEF00DULL;
                  bool all_match = true;
                  std::size_t first_bad = 0;
                  for (std::size_t n = 0; n <= 40; ++n) {
                      std::vector<std::uint64_t> batch(n);
                      counter_u64_batch(key, 1000, batch);
                      for (std::size_t i = 0; i < n; ++i) {
                          if (batch[i] != counter_u64(key, 1000 + i)) {
                              all_match = false;
                              first_bad = n;
                              break;
                          }
                      }
                      if (!all_match) {
                          break;
                      }
                  }
                  t.expect(all_match,
                           std::format("counter_u64_batch == scalar for all n in [0,40] "
                                       "(first mismatch at n={})",
                                       first_bad));

                  // A large fill exercises many vector groups at once.
                  constexpr std::size_t big = 10007;  // prime: guarantees a non-zero tail
                  std::vector<std::uint64_t> b(big);
                  counter_u64_batch(0x0123456789ABCDEFULL, 0, b);
                  bool big_match = true;
                  for (std::size_t i = 0; i < big; ++i) {
                      if (b[i] != counter_u64(0x0123456789ABCDEFULL, i)) {
                          big_match = false;
                          break;
                      }
                  }
                  t.expect(big_match, "counter_u64_batch matches scalar over a 10007-wide fill");
              })
        .test("rng_fill_u64_matches_sequential_next",
              [](TestContext& t) {
                  // Rng::fill_u64 must equal the same count of next_u64() calls AND leave the
                  // counter at the same position, so batched and scalar draws interleave freely.
                  auto a = Rng::seeded(7);
                  auto b = Rng::seeded(7);
                  constexpr std::size_t k = 37;  // not a multiple of 8: exercises the tail path
                  std::vector<std::uint64_t> filled(k);
                  a.fill_u64(filled);
                  bool match = true;
                  for (std::size_t i = 0; i < k; ++i) {
                      if (filled[i] != b.next_u64()) {
                          match = false;
                          break;
                      }
                  }
                  t.expect(match, "fill_u64(k) equals k sequential next_u64() draws");
                  t.expect_eq(a.counter(), b.counter(),
                              "fill_u64 advances the counter exactly k positions");
                  // The two streams remain in lock-step for subsequent draws.
                  t.expect_eq(a.next_u64(), b.next_u64(),
                              "streams stay aligned after a batched fill");
              })
        .run();
}
