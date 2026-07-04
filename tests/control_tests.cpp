// Tests for nimblecas.control: linear control systems over the exact rationals.
// @author Olumuyiwa Oluwasanmi
//
// Exact where the mathematics is exact (TF/SS algebra, TF<->SS, controllability /
// observability, Routh-Hurwitz and Schur stability, the bilinear transform) and
// numerical only for the Bode magnitude/phase, which is checked to a tolerance.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.complex;
import nimblecas.control;
import nimblecas.testing;

using nimblecas::bilinear_c2d;
using nimblecas::bode;
using nimblecas::bode_shard;
using nimblecas::BodePoint;
using nimblecas::Complex;
using nimblecas::evaluate_exact;
using nimblecas::hurwitz_minors;
using nimblecas::is_hurwitz_stable;
using nimblecas::is_spd;
using nimblecas::is_robustly_stable;
using nimblecas::is_stable_continuous;
using nimblecas::is_stable_discrete;
using nimblecas::is_lyapunov_stable;
using nimblecas::kharitonov_polynomials;
using nimblecas::lyapunov_solve;
using nimblecas::logspace;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::nyquist;
using nimblecas::nyquist_criterion;
using nimblecas::NyquistPoint;
using nimblecas::parallel_bode;
using nimblecas::parallel_nyquist;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::ss_to_tf;
using nimblecas::StateSpace;
using nimblecas::stability_margins;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
using nimblecas::tf_to_ss;
using nimblecas::TransferFunction;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}
[[nodiscard]] auto rq(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// A RationalPoly from integer coefficients, ascending (index i is the coeff of x^i).
[[nodiscard]] auto rp(std::vector<std::int64_t> coeffs) -> RationalPoly {
    std::vector<Rational> r;
    r.reserve(coeffs.size());
    for (const std::int64_t v : coeffs) {
        r.push_back(ri(v));
    }
    return RationalPoly::from_coeffs(std::move(r));
}

// A RationalPoly from rational coefficients given as {num, den} pairs, ascending.
[[nodiscard]] auto rpq(std::vector<std::pair<std::int64_t, std::int64_t>> coeffs) -> RationalPoly {
    std::vector<Rational> r;
    r.reserve(coeffs.size());
    for (const auto& [n, d] : coeffs) {
        r.push_back(rq(n, d));
    }
    return RationalPoly::from_coeffs(std::move(r));
}

[[nodiscard]] auto TF(RationalPoly num, RationalPoly den) -> TransferFunction {
    return TransferFunction::make(std::move(num), std::move(den)).value();
}

// A Matrix from integer rows (low-index row first).
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(ri(v));
        }
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.control")
        .test("poles_second_order_rational",
              [](TestContext& t) {
                  // H = 1/(s^2 + 3s + 2): poles at s = -1 and s = -2, fully rational.
                  const auto h = TF(rp({1}), rp({2, 3, 1}));
                  auto p = h.poles();
                  t.expect(p.has_value(), "poles computed");
                  if (p) {
                      t.expect(p->fully_extracted(), "spectrum fully rational (unextracted == 0)");
                      t.expect(p->rational.size() == 2, "two distinct rational poles");
                      bool has_m1 = false;
                      bool has_m2 = false;
                      for (const auto& [value, mult] : p->rational) {
                          if (value == ri(-1) && mult == 1) has_m1 = true;
                          if (value == ri(-2) && mult == 1) has_m2 = true;
                      }
                      t.expect(has_m1 && has_m2, "poles are exactly {-1, -2}");
                  }
              })
        .test("poles_irrational_not_fully_extracted",
              [](TestContext& t) {
                  // H = 1/(s^2 - 2): poles at +/- sqrt(2), irrational -> not extracted over Q.
                  const auto h = TF(rp({1}), rp({-2, 0, 1}));
                  auto p = h.poles();
                  t.expect(p.has_value(), "poles computed");
                  if (p) {
                      t.expect(p->rational.empty(), "no rational poles");
                      t.expect(!p->fully_extracted() && p->unextracted == 2,
                               "two poles reported as not fully extracted");
                  }
              })
        .test("dc_gain_and_properness",
              [](TestContext& t) {
                  // H = (2s + 4)/(s^2 + 3s + 2): H(0) = 4/2 = 2, strictly proper.
                  const auto h = TF(rp({4, 2}), rp({2, 3, 1}));
                  auto g = h.dc_gain();
                  t.expect(g.has_value() && g.value_or(ri(0)) == ri(2), "dc gain is 2");
                  t.expect(h.is_proper() && h.is_strictly_proper(), "strictly proper");
                  t.expect(h.relative_degree() == 1, "relative degree 1");
              })
        .test("tf_to_ss_then_ss_to_tf_roundtrip_strictly_proper",
              [](TestContext& t) {
                  // H = (s + 1)/(s^2 + 3s + 2). Round-trip must be exactly equivalent.
                  const auto h = TF(rp({1, 1}), rp({2, 3, 1}));
                  auto ss = tf_to_ss(h);
                  t.expect(ss.has_value(), "tf_to_ss succeeds");
                  if (ss) {
                      t.expect(ss->n_states() == 2, "two states");
                      auto back = ss_to_tf(*ss);
                      t.expect(back.has_value(), "ss_to_tf succeeds");
                      if (back) {
                          auto eq = h.equivalent(*back);
                          t.expect(eq.has_value() && eq.value_or(false),
                                   "round-trip is exactly equivalent");
                      }
                  }
              })
        .test("tf_to_ss_then_ss_to_tf_roundtrip_with_feedthrough",
              [](TestContext& t) {
                  // H = (s^2 + 1)/(s^2 + 3s + 2): proper but not strictly (feedthrough D = 1).
                  const auto h = TF(rp({1, 0, 1}), rp({2, 3, 1}));
                  auto ss = tf_to_ss(h);
                  t.expect(ss.has_value(), "tf_to_ss succeeds for a proper (non-strict) plant");
                  if (ss) {
                      t.expect(ss->d().at(0, 0) == ri(1), "feedthrough D = 1");
                      auto back = ss_to_tf(*ss);
                      t.expect(back.has_value(), "ss_to_tf succeeds");
                      if (back) {
                          auto eq = h.equivalent(*back);
                          t.expect(eq.has_value() && eq.value_or(false),
                                   "round-trip is exactly equivalent");
                      }
                  }
              })
        .test("tf_to_ss_rejects_improper",
              [](TestContext& t) {
                  // deg(num) > deg(den): not realisable without differentiators.
                  const auto h = TF(rp({0, 0, 1}), rp({1, 1}));
                  auto ss = tf_to_ss(h);
                  t.expect(!ss.has_value() && ss.error() == MathError::domain_error,
                           "improper plant => domain_error");
              })
        .test("controllability_kalman_rank",
              [](TestContext& t) {
                  // Controllable pair: A = [[0,1],[0,0]], B = [0,1]^T.
                  auto ctrb = StateSpace::make(mat({{0, 1}, {0, 0}}), mat({{0}, {1}}),
                                               mat({{1, 0}}), mat({{0}}));
                  t.expect(ctrb.has_value(), "state space assembled");
                  if (ctrb) {
                      auto ok = ctrb->is_controllable();
                      t.expect(ok.has_value() && ok.value_or(false),
                               "([[0,1],[0,0]], [0,1]) is controllable");
                  }
                  // Uncontrollable pair: A = I, B = [1,0]^T -> [B AB] has rank 1.
                  auto un = StateSpace::make(mat({{1, 0}, {0, 1}}), mat({{1}, {0}}),
                                             mat({{1, 0}}), mat({{0}}));
                  t.expect(un.has_value(), "state space assembled");
                  if (un) {
                      auto ok = un->is_controllable();
                      t.expect(ok.has_value() && !ok.value_or(true),
                               "(I, [1,0]) is NOT controllable");
                  }
              })
        .test("observability_kalman_rank",
              [](TestContext& t) {
                  // Observable pair: A = [[0,1],[0,0]], C = [1,0].
                  auto obs = StateSpace::make(mat({{0, 1}, {0, 0}}), mat({{0}, {1}}),
                                              mat({{1, 0}}), mat({{0}}));
                  t.expect(obs.has_value(), "state space assembled");
                  if (obs) {
                      auto ok = obs->is_observable();
                      t.expect(ok.has_value() && ok.value_or(false),
                               "([[0,1],[0,0]], C=[1,0]) is observable");
                  }
                  // Unobservable pair: A = I, C = [1,0] -> [C; CA] has rank 1.
                  auto un = StateSpace::make(mat({{1, 0}, {0, 1}}), mat({{0}, {1}}),
                                             mat({{1, 0}}), mat({{0}}));
                  t.expect(un.has_value(), "state space assembled");
                  if (un) {
                      auto ok = un->is_observable();
                      t.expect(ok.has_value() && !ok.value_or(true),
                               "(I, C=[1,0]) is NOT observable");
                  }
              })
        .test("routh_hurwitz_stable_vs_unstable",
              [](TestContext& t) {
                  // Stable: s^2 + 3s + 2 (roots -1, -2).
                  auto s1 = is_stable_continuous(TF(rp({1}), rp({2, 3, 1})));
                  t.expect(s1.has_value() && s1.value_or(false), "s^2+3s+2 is Hurwitz-stable");
                  // Unstable: s^2 - 1 (roots +/- 1).
                  auto s2 = is_stable_continuous(TF(rp({1}), rp({-1, 0, 1})));
                  t.expect(s2.has_value() && !s2.value_or(true), "s^2-1 is not stable");
                  // Marginal: s^2 + 1 (roots +/- i) -> not asymptotically stable.
                  auto s3 = is_stable_continuous(TF(rp({1}), rp({1, 0, 1})));
                  t.expect(s3.has_value() && !s3.value_or(true), "s^2+1 (imag axis) is not stable");
              })
        .test("discrete_schur_stability",
              [](TestContext& t) {
                  // Pole at z = 1/2 (inside the unit circle): stable.
                  auto in = is_stable_discrete(TF(rp({1}), rpq({{-1, 2}, {1, 1}})));
                  t.expect(in.has_value() && in.value_or(false), "pole at 1/2 is Schur-stable");
                  // Pole at z = 2 (outside the unit circle): not stable.
                  auto out = is_stable_discrete(TF(rp({1}), rp({-2, 1})));
                  t.expect(out.has_value() && !out.value_or(true), "pole at 2 is not Schur-stable");
                  // Boundary pole at z = -1: not strictly inside -> not stable.
                  auto edge = is_stable_discrete(TF(rp({1}), rp({1, 1})));
                  t.expect(edge.has_value() && !edge.value_or(true),
                           "pole at -1 (unit circle) is not Schur-stable");
              })
        .test("series_and_feedback_algebra",
              [](TestContext& t) {
                  // Series of 1/s and 1/(s+1) equals 1/(s^2 + s).
                  const auto g1 = TF(rp({1}), rp({0, 1}));      // 1/s
                  const auto g2 = TF(rp({1}), rp({1, 1}));      // 1/(s+1)
                  auto cascade = g1.series(g2);
                  t.expect(cascade.has_value(), "series computed");
                  if (cascade) {
                      auto eq = cascade->equivalent(TF(rp({1}), rp({0, 1, 1})));
                      t.expect(eq.has_value() && eq.value_or(false), "1/s * 1/(s+1) = 1/(s^2+s)");
                  }
                  // Unity negative feedback of G = 1/s: G/(1+G) = 1/(s+1).
                  auto cl = g1.unity_feedback();
                  t.expect(cl.has_value(), "unity feedback computed");
                  if (cl) {
                      auto eq = cl->equivalent(TF(rp({1}), rp({1, 1})));
                      t.expect(eq.has_value() && eq.value_or(false),
                               "(1/s)/(1 + 1/s) = 1/(s+1)");
                  }
                  // Parallel of 1/s and 1/(s+1) = (2s+1)/(s^2+s).
                  auto par = g1.parallel(g2);
                  t.expect(par.has_value(), "parallel computed");
                  if (par) {
                      auto eq = par->equivalent(TF(rp({1, 2}), rp({0, 1, 1})));
                      t.expect(eq.has_value() && eq.value_or(false),
                               "1/s + 1/(s+1) = (2s+1)/(s^2+s)");
                  }
              })
        .test("bode_magnitude_and_phase_numeric",
              [](TestContext& t) {
                  // H = 1/(s+1). At omega = 1: |H(i)| = 1/sqrt(2) -> -3.0103 dB, phase -45 deg.
                  const auto h = TF(rp({1}), rp({1, 1}));
                  const std::array<double, 1> w{1.0};
                  const auto data = bode(h, w);
                  t.expect(data.size() == 1, "one Bode sample");
                  if (!data.empty()) {
                      t.expect(std::abs(data[0].magnitude_db - (-3.010299957)) < 1e-6,
                               "magnitude ~ -3.0103 dB at omega=1");
                      t.expect(std::abs(data[0].phase_deg - (-45.0)) < 1e-6,
                               "phase ~ -45 deg at omega=1");
                  }
              })
        .test("evaluate_exact_gaussian_rational",
              [](TestContext& t) {
                  // H = 1/(s+1) at s = i: exact value (1 - i)/2, staying in Q + Qi.
                  const auto h = TF(rp({1}), rp({1, 1}));
                  auto v = evaluate_exact(h, Complex::i());
                  t.expect(v.has_value(), "exact evaluation succeeds");
                  if (v) {
                      t.expect(v->real() == rq(1, 2) && v->imag() == rq(-1, 2),
                               "H(i) = 1/2 - (1/2) i exactly");
                  }
              })
        .test("bilinear_tustin_transform",
              [](TestContext& t) {
                  // H(s) = 1/(s+1), T = 2 (=> 2/T = 1), s = (z-1)/(z+1).
                  // H(z) = (z+1)/(2z).
                  const auto h = TF(rp({1}), rp({1, 1}));
                  auto hz = bilinear_c2d(h, ri(2));
                  t.expect(hz.has_value(), "bilinear transform computed");
                  if (hz) {
                      auto eq = hz->equivalent(TF(rp({1, 1}), rp({0, 2})));  // (z+1)/(2z)
                      t.expect(eq.has_value() && eq.value_or(false),
                               "Tustin of 1/(s+1) with T=2 is (z+1)/(2z)");
                  }
              })
        .test("hurwitz_determinant_agrees_with_routh",
              [](TestContext& t) {
                  // The Hurwitz-minor (determinant) test must agree with the Routh-based
                  // is_stable_continuous on the same denominator, stable and unstable.
                  const std::array<RationalPoly, 4> polys{
                      rp({2, 3, 1}),      // s^2 + 3s + 2  (stable: roots -1, -2)
                      rp({-1, 0, 1}),     // s^2 - 1       (unstable: roots +/- 1)
                      rp({1, 3, 3, 1}),   // s^3+3s^2+3s+1 (stable: triple root -1)
                      rp({8, 2, 1, 1})};  // s^3+s^2+2s+8  (unstable: a2*a1 < a0)
                  for (const auto& p : polys) {
                      auto det_stable = is_hurwitz_stable(p);
                      auto routh_stable = is_stable_continuous(TF(rp({1}), p));
                      t.expect(det_stable.has_value() && routh_stable.has_value(),
                               "both stability tests computed");
                      t.expect(det_stable.value_or(false) == routh_stable.value_or(true),
                               "Hurwitz determinant verdict matches Routh-Hurwitz");
                  }
                  // The leading minors of a stable poly are all strictly positive.
                  auto minors = hurwitz_minors(rp({2, 3, 1}));
                  t.expect(minors.has_value() && minors->size() == 2, "two minors for degree 2");
                  if (minors) {
                      bool all_pos = true;
                      for (const auto& d : *minors) {
                          if (!(d.numerator() > 0)) all_pos = false;
                      }
                      t.expect(all_pos, "all leading principal minors > 0 for a stable poly");
                  }
              })
        .test("kharitonov_robust_interval_stability",
              [](TestContext& t) {
                  // Robustly stable family (degree 3, leading coeff fixed at 1):
                  // a0 in [1,2], a1 in [2,4], a2 in [2,4], a3 = 1.
                  const std::vector<Rational> lo_ok{ri(1), ri(2), ri(2), ri(1)};
                  const std::vector<Rational> hi_ok{ri(2), ri(4), ri(4), ri(1)};
                  auto robust = is_robustly_stable(lo_ok, hi_ok);
                  t.expect(robust.has_value() && robust.value_or(false),
                           "interval family is robustly stable (all four Kharitonov polys)");
                  auto four = kharitonov_polynomials(lo_ok, hi_ok);
                  t.expect(four.has_value() && four->size() == 4, "four Kharitonov polynomials");

                  // Not robustly stable: a1 ranges over [-1, 1], so one Kharitonov vertex
                  // has a negative coefficient and is not Hurwitz.
                  const std::vector<Rational> lo_bad{ri(1), ri(-1), ri(2), ri(1)};
                  const std::vector<Rational> hi_bad{ri(1), ri(1), ri(2), ri(1)};
                  auto not_robust = is_robustly_stable(lo_bad, hi_bad);
                  t.expect(not_robust.has_value() && !not_robust.value_or(true),
                           "interval family with a1 in [-1,1] is NOT robustly stable");
              })
        .test("lyapunov_positive_definite_matches_stability",
              [](TestContext& t) {
                  // A = diag(-1, -2) is Hurwitz: A^T P + P A = -I has P = diag(1/2, 1/4) > 0.
                  const auto a = mat({{-1, 0}, {0, -2}});
                  auto p = lyapunov_solve(a);
                  t.expect(p.has_value(), "Lyapunov equation solved exactly");
                  if (p) {
                      t.expect(p->at(0, 0) == rq(1, 2) && p->at(1, 1) == rq(1, 4) &&
                                   p->at(0, 1) == ri(0) && p->at(1, 0) == ri(0),
                               "P = diag(1/2, 1/4) exactly");
                      auto pd = is_spd(*p);
                      t.expect(pd.has_value() && pd.value_or(false), "P is positive definite");
                  }
                  auto lyap = is_lyapunov_stable(a);
                  t.expect(lyap.has_value() && lyap.value_or(false),
                           "Lyapunov verdict: diag(-1,-2) is stable");
                  // Cross-check against the state-space Routh-Hurwitz verdict.
                  auto ss = StateSpace::make(a, mat({{1}, {0}}), mat({{1, 0}}), mat({{0}}));
                  t.expect(ss.has_value(), "state space assembled");
                  if (ss) {
                      auto routh = ss->is_asymptotically_stable();
                      t.expect(routh.has_value() &&
                                   routh.value_or(false) == lyap.value_or(false),
                               "Lyapunov agrees with Routh-Hurwitz");
                  }
                  // Unstable A = diag(1, -2): P exists but is not positive definite.
                  auto un = is_lyapunov_stable(mat({{1, 0}, {0, -2}}));
                  t.expect(un.has_value() && !un.value_or(true),
                           "diag(1,-2) is not Lyapunov-stable");
                  // Rotation A = [[0,-1],[1,0]] (eigenvalues +/- i): singular Kronecker sum.
                  const auto rot = mat({{0, -1}, {1, 0}});
                  auto sing = lyapunov_solve(rot);
                  t.expect(!sing.has_value() && sing.error() == MathError::domain_error,
                           "purely imaginary spectrum => singular Lyapunov operator");
                  auto rot_stable = is_lyapunov_stable(rot);
                  t.expect(rot_stable.has_value() && !rot_stable.value_or(true),
                           "rotation is not Lyapunov-stable");
              })
        .test("nyquist_criterion_simple_loop",
              [](TestContext& t) {
                  const auto grid = logspace(1e-2, 1e2, 2000);
                  // Stable loop: L = 1/(s+1). No open-loop RHP poles, no encirclement of -1.
                  auto r1 = nyquist_criterion(TF(rp({1}), rp({1, 1})), grid);
                  t.expect(r1.has_value(), "Nyquist result computed");
                  if (r1) {
                      t.expect(r1->open_loop_rhp_poles == 0 && r1->p_exact,
                               "P = 0 (exact) for 1/(s+1)");
                      t.expect(r1->encirclements == 0 && r1->closed_loop_stable,
                               "no encirclement => closed loop stable");
                  }
                  // Unstable loop: L = 16/(s+1)^3 encircles -1 (|L| = 2 at the -180 crossing).
                  auto r2 = nyquist_criterion(TF(rp({16}), rp({1, 3, 3, 1})), grid);
                  t.expect(r2.has_value(), "Nyquist result computed");
                  if (r2) {
                      t.expect(r2->encirclements != 0 && !r2->closed_loop_stable,
                               "encirclement of -1 => closed loop unstable");
                  }
              })
        .test("gain_and_phase_margins_numeric",
              [](TestContext& t) {
                  // L = 2/(s+1)^3. Known margins: GM = 4 (12.04 dB) at omega_pc = sqrt(3),
                  // PM ~ 67.64 deg at omega_gc ~ 0.7664.
                  const auto grid = logspace(1e-2, 1e2, 4000);
                  const auto m = stability_margins(TF(rp({2}), rp({1, 3, 3, 1})), grid);
                  t.expect(m.has_gain_margin && m.has_phase_margin, "both crossovers found");
                  t.expect(std::abs(m.gain_margin_db - 12.0412) < 0.1,
                           "gain margin ~ 12.04 dB");
                  t.expect(std::abs(m.phase_crossover - 1.73205) < 0.02,
                           "phase crossover ~ sqrt(3)");
                  t.expect(std::abs(m.phase_margin - 67.6418) < 0.3, "phase margin ~ 67.64 deg");
                  t.expect(std::abs(m.gain_crossover - 0.76642) < 0.01,
                           "gain crossover ~ 0.7664");
              })
        .test("parallel_bode_matches_serial_bit_identical",
              [](TestContext& t) {
                  // parallel_bode / parallel_nyquist are order-preserving maps over independent
                  // per-ω evaluations: they must equal the serial sweep element-for-element and
                  // BIT-for-BIT (same double arithmetic), regardless of thread count. Use a grid
                  // well past the parallel grain size so the parallel path is actually exercised.
                  const auto h = TF(rp({1, 2}), rp({2, 3, 1}));  // (2s + 1)/(s^2 + 3s + 2)
                  const auto grid = logspace(1e-3, 1e3, 5000);

                  const auto serial_b = bode(h, grid);
                  const auto par_b = parallel_bode(h, grid);
                  t.expect(par_b.size() == serial_b.size(),
                           "parallel_bode size matches serial bode");
                  bool bode_identical = par_b.size() == serial_b.size();
                  for (std::size_t i = 0; i < par_b.size() && bode_identical; ++i) {
                      // Exact equality: same ω order, and each field the identical double bit
                      // pattern (NaN-free here because none of these ω hit an imaginary-axis pole).
                      if (par_b[i].omega != serial_b[i].omega ||
                          par_b[i].magnitude_db != serial_b[i].magnitude_db ||
                          par_b[i].phase_deg != serial_b[i].phase_deg) {
                          bode_identical = false;
                      }
                  }
                  t.expect(bode_identical, "parallel_bode == serial bode element-for-element");

                  const auto serial_n = nyquist(h, grid);
                  const auto par_n = parallel_nyquist(h, grid);
                  t.expect(par_n.size() == serial_n.size(),
                           "parallel_nyquist size matches serial nyquist");
                  bool nyq_identical = par_n.size() == serial_n.size();
                  for (std::size_t i = 0; i < par_n.size() && nyq_identical; ++i) {
                      if (par_n[i].omega != serial_n[i].omega ||
                          par_n[i].re != serial_n[i].re || par_n[i].im != serial_n[i].im) {
                          nyq_identical = false;
                      }
                  }
                  t.expect(nyq_identical,
                           "parallel_nyquist == serial nyquist element-for-element");
              })
        .test("bode_shard_reduction_reconstructs_full_sweep",
              [](TestContext& t) {
                  // Distributed reduction: concatenating the contiguous shards IN SHARD ORDER
                  // must reconstruct the full serial bode sweep exactly, for any shard count
                  // (partition independence + ordering). 4001 points -> uneven blocks, so the
                  // balanced-partition arithmetic is exercised, not just clean divisions.
                  const auto h = TF(rp({1, 2}), rp({2, 3, 1}));
                  const auto grid = logspace(1e-2, 1e2, 4001);
                  const auto full = bode(h, grid);

                  for (const std::size_t num_shards : {std::size_t{1}, std::size_t{2},
                                                       std::size_t{3}}) {
                      std::vector<BodePoint> reduced;
                      reduced.reserve(full.size());
                      for (std::size_t s = 0; s < num_shards; ++s) {
                          const auto part = bode_shard(h, grid, s, num_shards);
                          reduced.insert(reduced.end(), part.begin(), part.end());
                      }
                      t.expect(reduced.size() == full.size(),
                               "concatenated shards cover every frequency exactly once");
                      bool identical = reduced.size() == full.size();
                      for (std::size_t i = 0; i < reduced.size() && identical; ++i) {
                          if (reduced[i].omega != full[i].omega ||
                              reduced[i].magnitude_db != full[i].magnitude_db ||
                              reduced[i].phase_deg != full[i].phase_deg) {
                              identical = false;
                          }
                      }
                      t.expect(identical,
                               "bode_shard reduction is bit-identical to the full serial sweep");
                  }

                  // Out-of-range / empty-work shards contribute nothing.
                  t.expect(bode_shard(h, grid, 0, 0).empty(), "num_shards == 0 => empty shard");
                  t.expect(bode_shard(h, grid, 3, 3).empty(),
                           "shard_index >= num_shards => empty shard");
              })
        .run();
}
