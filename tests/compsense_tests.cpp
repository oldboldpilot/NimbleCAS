// Tests for nimblecas.compsense: exact basis pursuit + numerical OMP/CoSaMP/IHT.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.compsense;
import nimblecas.testing;

using nimblecas::basis_pursuit;
using nimblecas::coherence_guarantees_recovery;
using nimblecas::cosamp;
using nimblecas::iterative_hard_thresholding;
using nimblecas::MathError;
using nimblecas::mutual_coherence;
using nimblecas::mutual_coherence_squared;
using nimblecas::orthogonal_matching_pursuit;
using nimblecas::Rational;
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
        .run();
}
