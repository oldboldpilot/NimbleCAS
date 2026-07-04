// Tests for nimblecas.compsense: exact basis pursuit + numerical OMP/CoSaMP/IHT.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.compsense;
import nimblecas.testing;

using nimblecas::basis_pursuit;
using nimblecas::basis_pursuit_batch_shard;
using nimblecas::coherence_guarantees_recovery;
using nimblecas::concat_shards;
using nimblecas::cosamp;
using nimblecas::iterative_hard_thresholding;
using nimblecas::MathError;
using nimblecas::mutual_coherence;
using nimblecas::mutual_coherence_squared;
using nimblecas::orthogonal_matching_pursuit;
using nimblecas::parallel_basis_pursuit_batch;
using nimblecas::parallel_omp_batch;
using nimblecas::Rational;
using nimblecas::recover_batch_shard;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}
[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}
[[nodiscard]] auto close(double got, double expected) -> bool {
    return std::abs(got - expected) < 1e-6;
}

// Exact (bit-identical) comparison of two per-signal recovery Results: same success/
// failure, same error, and — crucially for the determinism claim — the SAME bits, so
// we use == on the doubles, not a tolerance.
[[nodiscard]] auto same(const Result<std::vector<double>>& a,
                        const Result<std::vector<double>>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    if (!a.has_value()) {
        return a.error() == b.error();
    }
    return std::ranges::equal(*a, *b);
}
[[nodiscard]] auto same(const Result<std::vector<Rational>>& a,
                        const Result<std::vector<Rational>>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    if (!a.has_value()) {
        return a.error() == b.error();
    }
    return std::ranges::equal(*a, *b);
}

// The shared 4x5 test dictionary (columns e0..e3 and a4 = (1/2,1/2,1/2,1/2)).
[[nodiscard]] auto dict_4x5() -> std::vector<double> {
    return std::vector<double>{1, 0, 0, 0, 0.5,   //
                               0, 1, 0, 0, 0.5,    //
                               0, 0, 1, 0, 0.5,    //
                               0, 0, 0, 1, 0.5};   //
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.compsense")
        .test("basis_pursuit_exact_two_sparse",
              [](TestContext& t) {
                  // Atoms are the columns of A: a0=(1,0), a1=(1,1), a2=(0,1).
                  // With b = (2, 3) the unique min-L1 solution is the 2-sparse x=(0,2,1);
                  // basis pursuit must return it EXACTLY (no rounding).
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  const std::vector<Rational> b{ri(2), ri(3)};
                  auto x = basis_pursuit(A, b);
                  t.expect(x.has_value(), "basis pursuit succeeds");
                  t.expect(x.has_value() && x->size() == 3, "solution has one entry per atom");
                  t.expect(x.has_value() && (*x)[0] == ri(0), "x0 = 0 exactly");
                  t.expect(x.has_value() && (*x)[1] == ri(2), "x1 = 2 exactly");
                  t.expect(x.has_value() && (*x)[2] == ri(1), "x2 = 1 exactly");
              })
        .test("basis_pursuit_exact_fraction",
              [](TestContext& t) {
                  // A genuinely fractional exact minimiser: b = (1/2, 5/2) => x=(0, 1/2, 2).
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  const std::vector<Rational> b{rat(1, 2), rat(5, 2)};
                  auto x = basis_pursuit(A, b);
                  t.expect(x.has_value(), "basis pursuit succeeds on rational RHS");
                  t.expect(x.has_value() && (*x)[0] == ri(0), "x0 = 0 exactly");
                  t.expect(x.has_value() && (*x)[1] == rat(1, 2), "x1 = 1/2 exactly");
                  t.expect(x.has_value() && (*x)[2] == ri(2), "x2 = 2 exactly");
              })
        .test("basis_pursuit_zero_rhs",
              [](TestContext& t) {
                  // b = 0 => the exact minimiser is the zero vector.
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  const std::vector<Rational> b{ri(0), ri(0)};
                  auto x = basis_pursuit(A, b);
                  t.expect(x.has_value() && x->size() == 3, "zero RHS solves");
                  t.expect(x.has_value() && (*x)[0] == ri(0) && (*x)[1] == ri(0) &&
                               (*x)[2] == ri(0),
                           "x is exactly zero");
              })
        .test("omp_recovers_sparse_signal",
              [](TestContext& t) {
                  // 4x5 incoherent dictionary: e0..e3 plus a4 = (1/2,1/2,1/2,1/2) (unit norm,
                  // coherence 1/2 with each e_i). Signal x = 3 e1 - 2 e3, b = A x.
                  const std::vector<double> A{1, 0, 0, 0, 0.5,   //
                                              0, 1, 0, 0, 0.5,   //
                                              0, 0, 1, 0, 0.5,   //
                                              0, 0, 0, 1, 0.5};  //
                  const std::vector<double> b{0.0, 3.0, 0.0, -2.0};
                  auto x = orthogonal_matching_pursuit(A, 4, 5, b, 2, 1e-9);
                  t.expect(x.has_value() && x->size() == 5, "OMP returns 5 coefficients");
                  t.expect(x.has_value() && close((*x)[1], 3.0), "coeff on e1 ~ 3");
                  t.expect(x.has_value() && close((*x)[3], -2.0), "coeff on e3 ~ -2");
                  t.expect(x.has_value() && close((*x)[0], 0.0) && close((*x)[2], 0.0) &&
                               close((*x)[4], 0.0),
                           "off-support coefficients ~ 0");
              })
        .test("cosamp_recovers_sparse_signal",
              [](TestContext& t) {
                  const std::vector<double> A{1, 0, 0, 0, 0.5,   //
                                              0, 1, 0, 0, 0.5,   //
                                              0, 0, 1, 0, 0.5,   //
                                              0, 0, 0, 1, 0.5};  //
                  const std::vector<double> b{0.0, 3.0, 0.0, -2.0};
                  auto x = cosamp(A, 4, 5, b, 2, 50, 1e-9);
                  t.expect(x.has_value() && close((*x)[1], 3.0), "CoSaMP coeff on e1 ~ 3");
                  t.expect(x.has_value() && close((*x)[3], -2.0), "CoSaMP coeff on e3 ~ -2");
              })
        .test("iht_recovers_on_orthonormal_dict",
              [](TestContext& t) {
                  // Orthonormal (identity) dictionary: one gradient step with unit step
                  // recovers the exact 2-sparse signal.
                  const std::vector<double> A{1, 0, 0, 0,   //
                                              0, 1, 0, 0,   //
                                              0, 0, 1, 0,   //
                                              0, 0, 0, 1};  //
                  const std::vector<double> b{0.0, 3.0, 0.0, -2.0};
                  auto x = iterative_hard_thresholding(A, 4, 4, b, 2, 1.0, 500, 1e-12);
                  t.expect(x.has_value() && close((*x)[1], 3.0), "IHT coeff on e1 ~ 3");
                  t.expect(x.has_value() && close((*x)[3], -2.0), "IHT coeff on e3 ~ -2");
              })
        .test("mutual_coherence_squared_hand_value",
              [](TestContext& t) {
                  // Columns (1,0), (1,1), (0,1): pairwise squared coherence is exactly 1/2.
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  auto mu2 = mutual_coherence_squared(A);
                  t.expect(mu2.has_value(), "squared coherence computes");
                  t.expect(mu2.has_value() && *mu2 == rat(1, 2),
                           "mu^2 = 1/2 exactly (squared form)");
                  // The numerical coherence is the sqrt, ~ 0.70710678.
                  const std::vector<double> Ad{1, 1, 0, 0, 1, 1};
                  auto mu = mutual_coherence(Ad, 2, 3);
                  t.expect(mu.has_value() && close(*mu, std::sqrt(0.5)),
                           "numerical mu ~ sqrt(1/2)");
              })
        .test("coherence_recovery_bound_succeed_vs_fail",
              [](TestContext& t) {
                  // With mu^2 = 1/2, the sufficient bound k < (1 + 1/mu)/2 holds for k=1
                  // but not for k=2 (it is sufficient, not necessary).
                  const Rational mu2 = rat(1, 2);
                  auto ok1 = coherence_guarantees_recovery(mu2, 1);
                  t.expect(ok1.has_value() && *ok1 == true, "k=1 is guaranteed to recover");
                  auto ok2 = coherence_guarantees_recovery(mu2, 2);
                  t.expect(ok2.has_value() && *ok2 == false, "k=2 is not guaranteed");
                  // A very incoherent dictionary (mu^2 = 1/100) guarantees k=3.
                  auto ok3 = coherence_guarantees_recovery(rat(1, 100), 3);
                  t.expect(ok3.has_value() && *ok3 == true, "low coherence guarantees k=3");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // Ragged A into basis pursuit.
                  const std::vector<std::vector<Rational>> ragged{{ri(1), ri(1), ri(0)},
                                                                 {ri(1)}};
                  const std::vector<Rational> b2{ri(1), ri(1)};
                  auto r1 = basis_pursuit(ragged, b2);
                  t.expect(!r1.has_value() && r1.error() == MathError::domain_error,
                           "ragged A -> domain_error");

                  // b length disagreeing with the number of rows.
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                            {ri(0), ri(1), ri(1)}};
                  const std::vector<Rational> b_short{ri(1)};
                  auto r2 = basis_pursuit(A, b_short);
                  t.expect(!r2.has_value() && r2.error() == MathError::domain_error,
                           "b length mismatch -> domain_error");

                  // Coherence of a single-atom dictionary is undefined.
                  const std::vector<std::vector<Rational>> one_atom{{ri(1)}, {ri(0)}};
                  auto r3 = mutual_coherence_squared(one_atom);
                  t.expect(!r3.has_value() && r3.error() == MathError::domain_error,
                           "single atom -> domain_error");

                  // OMP with a data span whose length disagrees with rows*cols.
                  const std::vector<double> bad{1.0, 0.0, 0.0};  // not 4*5
                  const std::vector<double> bvec{0.0, 3.0, 0.0, -2.0};
                  auto r4 = orthogonal_matching_pursuit(bad, 4, 5, bvec, 2, 1e-9);
                  t.expect(!r4.has_value() && r4.error() == MathError::domain_error,
                           "OMP shape mismatch -> domain_error");
              })
        .test("parallel_omp_batch_equals_serial",
              [](TestContext& t) {
                  // One dictionary A, several signals recovered together. The parallel
                  // batch must be BIT-IDENTICAL to recovering each vector serially, and
                  // must preserve input order.
                  const std::vector<double> A = dict_4x5();
                  const std::vector<std::vector<double>> measurements{
                      {0.0, 3.0, 0.0, -2.0},        // 3 e1 - 2 e3
                      {5.0, 0.0, 0.0, 0.0},         // 5 e0
                      {0.0, 0.0, 0.0, 0.0},         // zero signal
                      {0.5, 0.5, 0.5, 0.5}};        // the shared atom a4
                  auto batch = parallel_omp_batch(A, 4, 5, measurements, 2, 1e-9);
                  t.expect(batch.size() == measurements.size(),
                           "batch has one result per measurement, in order");
                  bool all_match = batch.size() == measurements.size();
                  for (std::size_t k = 0; k < measurements.size(); ++k) {
                      auto serial = orthogonal_matching_pursuit(A, 4, 5, measurements[k], 2, 1e-9);
                      all_match = all_match && same(batch[k], serial);
                  }
                  t.expect(all_match, "each parallel result == serial result (bit-identical)");
                  // And it genuinely recovers: first signal is 3 e1 - 2 e3.
                  t.expect(batch[0].has_value() && close((*batch[0])[1], 3.0) &&
                               close((*batch[0])[3], -2.0),
                           "batch[0] recovers 3 e1 - 2 e3");
              })
        .test("parallel_basis_pursuit_batch_equals_serial",
              [](TestContext& t) {
                  // Exact-over-Q batch: columns (1,0),(1,1),(0,1); three RHS vectors.
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  const std::vector<std::vector<Rational>> measurements{
                      {ri(2), ri(3)},               // -> (0, 2, 1)
                      {rat(1, 2), rat(5, 2)},       // -> (0, 1/2, 2)
                      {ri(0), ri(0)}};              // -> (0, 0, 0)
                  auto batch = parallel_basis_pursuit_batch(A, measurements);
                  t.expect(batch.size() == 3, "batch has one exact result per RHS, in order");
                  bool all_match = batch.size() == 3;
                  for (std::size_t k = 0; k < measurements.size(); ++k) {
                      auto serial = basis_pursuit(A, measurements[k]);
                      all_match = all_match && same(batch[k], serial);
                  }
                  t.expect(all_match, "each parallel exact result == serial (exact over Q)");
                  t.expect(batch[1].has_value() && (*batch[1])[1] == rat(1, 2) &&
                               (*batch[1])[2] == ri(2),
                           "batch[1] recovers (0, 1/2, 2) exactly");
              })
        .test("omp_shard_reduction_reconstructs_full_batch",
              [](TestContext& t) {
                  const std::vector<double> A = dict_4x5();
                  const std::vector<std::vector<double>> measurements{
                      {0.0, 3.0, 0.0, -2.0}, {5.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0},  {0.5, 0.5, 0.5, 0.5},
                      {1.0, 0.0, 0.0, 0.0}};  // 5 signals -> uneven 2-way split (3 + 2)
                  const std::size_t total = measurements.size();
                  auto full = parallel_omp_batch(A, 4, 5, measurements, 2, 1e-9);

                  // num_shards = 1: the single shard is the whole batch.
                  auto s0 = recover_batch_shard(A, 4, 5, measurements, 0, 1, 2, 1e-9);
                  t.expect(s0.has_value() && s0->size() == total, "shard 0/1 covers all signals");
                  std::vector<std::vector<Result<std::vector<double>>>> one;
                  one.push_back(std::move(*s0));
                  auto joined1 = concat_shards<double>(total, 1, std::move(one));
                  bool ok1 = joined1.has_value() && joined1->size() == total;
                  for (std::size_t i = 0; ok1 && i < total; ++i) {
                      ok1 = ok1 && same((*joined1)[i], full[i]);
                  }
                  t.expect(ok1, "num_shards=1 reconstruction == full batch, in order");

                  // num_shards = 2: shard 0 owns {0,2,4}, shard 1 owns {1,3}.
                  auto sh0 = recover_batch_shard(A, 4, 5, measurements, 0, 2, 2, 1e-9);
                  auto sh1 = recover_batch_shard(A, 4, 5, measurements, 1, 2, 2, 1e-9);
                  t.expect(sh0.has_value() && sh0->size() == 3, "shard 0/2 owns indices 0,2,4");
                  t.expect(sh1.has_value() && sh1->size() == 2, "shard 1/2 owns indices 1,3");
                  std::vector<std::vector<Result<std::vector<double>>>> two;
                  two.push_back(std::move(*sh0));
                  two.push_back(std::move(*sh1));
                  auto joined2 = concat_shards<double>(total, 2, std::move(two));
                  bool ok2 = joined2.has_value() && joined2->size() == total;
                  for (std::size_t i = 0; ok2 && i < total; ++i) {
                      ok2 = ok2 && same((*joined2)[i], full[i]);
                  }
                  t.expect(ok2, "num_shards=2 reconstruction == full batch, in order");
              })
        .test("basis_pursuit_shard_reduction_reconstructs_full_batch",
              [](TestContext& t) {
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1), ri(0)},
                                                             {ri(0), ri(1), ri(1)}};
                  const std::vector<std::vector<Rational>> measurements{
                      {ri(2), ri(3)}, {rat(1, 2), rat(5, 2)}, {ri(0), ri(0)}, {ri(1), ri(1)}};
                  const std::size_t total = measurements.size();
                  auto full = parallel_basis_pursuit_batch(A, measurements);

                  auto sh0 = basis_pursuit_batch_shard(A, measurements, 0, 2);
                  auto sh1 = basis_pursuit_batch_shard(A, measurements, 1, 2);
                  t.expect(sh0.has_value() && sh0->size() == 2, "exact shard 0/2 owns 0,2");
                  t.expect(sh1.has_value() && sh1->size() == 2, "exact shard 1/2 owns 1,3");
                  std::vector<std::vector<Result<std::vector<Rational>>>> two;
                  two.push_back(std::move(*sh0));
                  two.push_back(std::move(*sh1));
                  auto joined = concat_shards<Rational>(total, 2, std::move(two));
                  bool ok = joined.has_value() && joined->size() == total;
                  for (std::size_t i = 0; ok && i < total; ++i) {
                      ok = ok && same((*joined)[i], full[i]);
                  }
                  t.expect(ok, "exact sharded reconstruction == full batch, in order");
              })
        .test("shard_and_concat_domain_errors",
              [](TestContext& t) {
                  const std::vector<double> A = dict_4x5();
                  const std::vector<std::vector<double>> measurements{{0.0, 3.0, 0.0, -2.0}};
                  auto e0 = recover_batch_shard(A, 4, 5, measurements, 0, 0, 2, 1e-9);
                  t.expect(!e0.has_value() && e0.error() == MathError::domain_error,
                           "num_shards=0 -> domain_error");
                  auto e1 = recover_batch_shard(A, 4, 5, measurements, 3, 2, 2, 1e-9);
                  t.expect(!e1.has_value() && e1.error() == MathError::domain_error,
                           "shard_index >= num_shards -> domain_error");
                  // concat with the wrong number of shards.
                  std::vector<std::vector<Result<std::vector<double>>>> empty;
                  auto e2 = concat_shards<double>(1, 2, std::move(empty));
                  t.expect(!e2.has_value() && e2.error() == MathError::domain_error,
                           "shards.size() != num_shards -> domain_error");
              })
        .run();
}
