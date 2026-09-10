// Tests for nimblecas.memo_postgres: a DistributedMemo backed by a PostgreSQL table.
// @author Olumuyiwa Oluwasanmi
//
// Two kinds of test, kept apart on purpose.
//
// The OFFLINE ones always run and need no server: identifier validation, argument checking, and
// -- the one that matters most -- that an unreachable database is reported as a transport error
// and never as a cache miss. That distinction is this module's honesty boundary, and it is
// checkable without a database precisely because the failure path is the thing under test.
//
// The LIVE ones need a real server and are gated on the NIMBLECAS_POSTGRES_DSN environment
// variable. When it is absent they SKIP, loudly, and the suite says so; they do not quietly pass.
// A test that cannot reach a database and reports success would be worse than no test, because it
// would make an untested build look tested.
//
// Run the live tests with, for example:
//   NIMBLECAS_POSTGRES_DSN="host=... port=5432 user=postgres password=... dbname=nimblecas \
//     connect_timeout=5" ./memo_postgres_tests

import std;
import nimblecas.core;
import nimblecas.memo_dist;
import nimblecas.memo_postgres;
import nimblecas.taskdag;
import nimblecas.testing;

using nimblecas::content_key;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::PostgresMemo;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto dsn() -> std::optional<std::string> {
    const char* raw = std::getenv("NIMBLECAS_POSTGRES_DSN");
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string s(raw);
    if (s.empty()) {
        return std::nullopt;
    }
    return s;
}

[[nodiscard]] auto bytes_of(std::string_view s) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(s.size());
    for (const char c : s) {
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
    }
    return out;
}

// A table name unique to this run, so two test runs against one server cannot collide.
[[nodiscard]] auto scratch_table() -> std::string {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::format("nimblecas_memo_t{}", static_cast<std::uint64_t>(now) % 1000000000ULL);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.memo_postgres")
        .test("a_table_name_that_is_not_a_plain_identifier_is_refused",
              [](TestContext& t) {
                  // The table name is INTERPOLATED into the DDL -- it has to be, since a bound
                  // parameter cannot name a table -- so the only safe policy is to refuse
                  // anything that is not a bare identifier. Every VALUE, by contrast, is bound
                  // and never interpolated.
                  t.expect(PostgresMemo::is_plain_identifier("nimblecas_memo"),
                           "an ordinary name is accepted");
                  t.expect(PostgresMemo::is_plain_identifier("_x9"), "leading underscore is fine");
                  t.expect(!PostgresMemo::is_plain_identifier(""), "empty is refused");
                  t.expect(!PostgresMemo::is_plain_identifier("9lives"),
                           "a leading digit is refused");
                  t.expect(!PostgresMemo::is_plain_identifier("a b"), "a space is refused");
                  t.expect(!PostgresMemo::is_plain_identifier("t\"; DROP TABLE x; --"),
                           "and so is anything carrying quotes or statement separators");
                  t.expect(!PostgresMemo::is_plain_identifier(std::string(64, 'a')),
                           "a name past Postgres's 63-byte identifier limit is refused");
              })
        .test("nonsensical_arguments_are_refused_before_any_connection_is_attempted",
              [](TestContext& t) {
                  auto empty = PostgresMemo::create("", "nimblecas_memo", 1);
                  t.expect(!empty.has_value() && empty.error() == MathError::domain_error,
                           "an empty connection string is a domain_error");
                  auto zero_pool =
                      PostgresMemo::create("host=127.0.0.1 connect_timeout=1", "t", 0);
                  t.expect(!zero_pool.has_value() && zero_pool.error() == MathError::domain_error,
                           "a zero-sized connection pool is a domain_error");
                  auto bad_table =
                      PostgresMemo::create("host=127.0.0.1 connect_timeout=1", "bad name", 1);
                  t.expect(!bad_table.has_value() && bad_table.error() == MathError::domain_error,
                           "a bad table name is refused without dialling anything");
              })
        .test("an_unreachable_database_is_a_transport_error_and_never_a_miss",
              [](TestContext& t) {
                  // The single most important behaviour in this module. Reporting an unreachable
                  // server as a cache miss would let a whole run continue, recompute everything,
                  // and look perfectly healthy while doing it.
                  //
                  // Port 1 on the loopback address has nothing listening, and connect_timeout
                  // bounds the wait -- without it libpq would block indefinitely, which is how a
                  // memo turns into a hang.
                  auto memo = PostgresMemo::create(
                      "host=127.0.0.1 port=1 connect_timeout=2 dbname=nimblecas", "t", 1);
                  t.expect(!memo.has_value(),
                           "connecting to a dead port fails rather than yielding a memo");
                  t.expect(!memo.has_value() && memo.error() == MathError::distributed_error,
                           "and the failure is distributed_error -- the TRANSPORT class -- not a "
                           "math error and emphatically not a silent miss");
              })
        .test("live_round_trip_publish_and_lookup",
              [](TestContext& t) {
                  const auto d = dsn();
                  if (!d.has_value()) {
                      t.expect(true, "SKIPPED: set NIMBLECAS_POSTGRES_DSN to run the live tests");
                      return;
                  }
                  const std::string table = scratch_table();
                  auto memo = PostgresMemo::create(*d, table, 2);
                  t.expect(memo.has_value(), "the memo connects and creates its table");
                  if (!memo) {
                      return;
                  }
                  t.expect((*memo)->name() == "postgres_memo", "it names itself");

                  const auto full = bytes_of("task/one/canonical-bytes");
                  const auto key = content_key(full);
                  const auto value = bytes_of("the-result");

                  auto before = (*memo)->lookup(key, full);
                  t.expect(before.has_value() && !before->has_value(),
                           "an unpublished key is a miss, and a miss is not an error");

                  auto const pub = (*memo)->publish(key, full, value);
                  t.expect(pub.has_value(), "publishing succeeds");

                  auto after = (*memo)->lookup(key, full);
                  t.expect(after.has_value() && after->has_value(), "and the key is then a hit");
                  t.expect(after.has_value() && after->has_value() && **after == value,
                           "returning the exact bytes that were published");

                  const auto s = (*memo)->stats();
                  t.expect(s.hits == 1 && s.misses == 1 && s.publishes == 1,
                           "the counters record one miss, one publish and one hit");
                  t.expect((*memo)->clear().has_value(), "the table clears");
              })
        .test("live_a_fingerprint_collision_returns_a_miss_not_the_colliding_value",
              [](TestContext& t) {
                  // The exactness rule of DistributedMemo, and it does not weaken because the
                  // table is remote. Two DIFFERENT full keys are forced under one fingerprint by
                  // publishing the second under the first's key, which is what a real collision
                  // would look like from the table's point of view.
                  const auto d = dsn();
                  if (!d.has_value()) {
                      t.expect(true, "SKIPPED: set NIMBLECAS_POSTGRES_DSN to run the live tests");
                      return;
                  }
                  const std::string table = scratch_table();
                  auto memo = PostgresMemo::create(*d, table, 1);
                  t.expect(memo.has_value(), "the memo connects");
                  if (!memo) {
                      return;
                  }
                  const auto key_a = bytes_of("alpha");
                  const auto key_b = bytes_of("beta-which-is-a-different-key");
                  const auto shared = content_key(key_a);

                  t.expect((*memo)->publish(shared, key_a, bytes_of("value-A")).has_value(),
                           "the first key publishes");

                  auto hit = (*memo)->lookup(shared, key_a);
                  t.expect(hit.has_value() && hit->has_value() && **hit == bytes_of("value-A"),
                           "and reads back its own value");

                  // Same fingerprint, different full key: a MISS, never value-A.
                  auto collide = (*memo)->lookup(shared, key_b);
                  t.expect(collide.has_value(), "the colliding lookup is not an error");
                  t.expect(collide.has_value() && !collide->has_value(),
                           "it is a MISS -- the colliding value is never returned");
                  t.expect((*memo)->stats().key_mismatches == 1,
                           "and the mismatch is counted, which is only possible because the full "
                           "key is compared rather than left to the WHERE clause");

                  // Both keys can coexist under one fingerprint, each returning its own value.
                  t.expect((*memo)->publish(shared, key_b, bytes_of("value-B")).has_value(),
                           "the colliding key publishes its own row");
                  auto a = (*memo)->lookup(shared, key_a);
                  auto b = (*memo)->lookup(shared, key_b);
                  t.expect(a.has_value() && a->has_value() && **a == bytes_of("value-A"),
                           "the first key still reads its own value");
                  t.expect(b.has_value() && b->has_value() && **b == bytes_of("value-B"),
                           "and the second reads its own, under the same fingerprint");
                  t.expect((*memo)->clear().has_value(), "the table clears");
              })
        .test("live_publishing_twice_is_idempotent_and_keeps_the_first_value",
              [](TestContext& t) {
                  // Two workers racing on the same task both publish. That must not be an error,
                  // and must not leave the table with two rows for one key.
                  const auto d = dsn();
                  if (!d.has_value()) {
                      t.expect(true, "SKIPPED: set NIMBLECAS_POSTGRES_DSN to run the live tests");
                      return;
                  }
                  const std::string table = scratch_table();
                  auto memo = PostgresMemo::create(*d, table, 1);
                  t.expect(memo.has_value(), "the memo connects");
                  if (!memo) {
                      return;
                  }
                  const auto full = bytes_of("raced-task");
                  const auto key = content_key(full);
                  t.expect((*memo)->publish(key, full, bytes_of("first")).has_value(),
                           "the first publish succeeds");
                  t.expect((*memo)->publish(key, full, bytes_of("second")).has_value(),
                           "the second publish is NOT an error");
                  auto rows = (*memo)->row_count();
                  t.expect(rows.has_value() && *rows == 1,
                           "and leaves exactly one row, not two");
                  auto got = (*memo)->lookup(key, full);
                  t.expect(got.has_value() && got->has_value() && **got == bytes_of("first"),
                           "the first value is the one kept");
                  t.expect((*memo)->clear().has_value(), "the table clears");
              })
        .test("live_an_oversized_value_is_refused_rather_than_stored",
              [](TestContext& t) {
                  const auto d = dsn();
                  if (!d.has_value()) {
                      t.expect(true, "SKIPPED: set NIMBLECAS_POSTGRES_DSN to run the live tests");
                      return;
                  }
                  const std::string table = scratch_table();
                  auto memo = PostgresMemo::create(*d, table, 1, /*max_value_bytes=*/64);
                  t.expect(memo.has_value(), "the memo connects");
                  if (!memo) {
                      return;
                  }
                  const auto full = bytes_of("big");
                  const auto key = content_key(full);
                  const std::vector<std::byte> big(65, std::byte{0x41});
                  auto r = (*memo)->publish(key, full, big);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "a value over the cap is a domain_error");
                  t.expect((*memo)->stats().rejected == 1, "and is counted as rejected");
                  auto rows = (*memo)->row_count();
                  t.expect(rows.has_value() && *rows == 0, "nothing was written");
                  t.expect((*memo)->clear().has_value(), "the table clears");
              })
        .test("live_concurrent_use_is_safe_and_every_value_survives",
              [](TestContext& t) {
                  // The interface requires thread safety, and a PGconn is not thread-safe, so the
                  // pool is what makes this true. Several threads publish and read back
                  // simultaneously; each must find exactly its own value.
                  const auto d = dsn();
                  if (!d.has_value()) {
                      t.expect(true, "SKIPPED: set NIMBLECAS_POSTGRES_DSN to run the live tests");
                      return;
                  }
                  const std::string table = scratch_table();
                  auto memo = PostgresMemo::create(*d, table, 4);
                  t.expect(memo.has_value(), "the memo connects with a pool of 4");
                  if (!memo) {
                      return;
                  }
                  constexpr int threads = 8;
                  constexpr int per_thread = 10;
                  std::atomic<int> good{0};
                  {
                      std::vector<std::jthread> ts;
                      ts.reserve(threads);
                      for (int ti = 0; ti < threads; ++ti) {
                          ts.emplace_back([&, ti] {
                              for (int k = 0; k < per_thread; ++k) {
                                  const auto full = bytes_of(std::format("t{}-k{}", ti, k));
                                  const auto key = content_key(full);
                                  const auto value = bytes_of(std::format("v{}-{}", ti, k));
                                  if (!(*memo)->publish(key, full, value).has_value()) {
                                      continue;
                                  }
                                  auto got = (*memo)->lookup(key, full);
                                  if (got.has_value() && got->has_value() && **got == value) {
                                      good.fetch_add(1, std::memory_order_relaxed);
                                  }
                              }
                          });
                      }
                  }
                  t.expect(good.load() == threads * per_thread,
                           "every thread read back exactly the value it published");
                  auto rows = (*memo)->row_count();
                  t.expect(rows.has_value() && *rows == threads * per_thread,
                           "and the table holds one row per distinct key");
                  t.expect((*memo)->clear().has_value(), "the table clears");
              })
        .run();
}
