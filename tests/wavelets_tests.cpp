// Tests for nimblecas.wavelets: Haar (exact over Q), Daubechies sum rules /
// orthogonality, biorthogonal PR, multi-level, SWT, lifting, and the CWT.
// @author Olumuyiwa Oluwasanmi
//
// Honesty is enforced by the choice of comparison: Haar and Haar-lifting are checked
// with EXACT rational equality (==), every other family with an explicit numerical
// tolerance.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.wavelets;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace wl = nimblecas::wavelets;

namespace {

// Absolute-tolerance comparison for the NUMERICAL families.
auto close(double got, double expected, double tol = 1e-9) -> bool {
    return std::abs(got - expected) < tol;
}

// Build an exact rational vector from integer numerators (denominator 1).
auto rats(std::initializer_list<std::int64_t> xs) -> std::vector<Rational> {
    std::vector<Rational> v;
    v.reserve(xs.size());
    for (std::int64_t x : xs) {
        v.push_back(Rational::from_int(x));
    }
    return v;
}

// Exact equality of two rational sequences.
auto equal_exact(std::span<const Rational> a, std::span<const Rational> b) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.wavelets")
        .test("haar_roundtrip_exact_over_Q",
              [](TestContext& t) {
                  // Non-dyadic rationals: reconstruction must still be EXACT in Q.
                  std::vector<Rational> x{*Rational::make(1, 3), *Rational::make(2, 5),
                                          Rational::from_int(7), *Rational::make(-4, 9),
                                          *Rational::make(11, 2), Rational::from_int(0),
                                          *Rational::make(-1, 7), *Rational::make(8, 3)};
                  auto fwd = wl::haar_dwt(x);
                  t.expect(fwd.has_value(), "haar_dwt succeeds on even length");
                  if (fwd) {
                      auto rec = wl::haar_idwt(fwd->approx, fwd->detail);
                      t.expect(rec.has_value(), "haar_idwt succeeds");
                      t.expect(rec.has_value() && equal_exact(*rec, x),
                               "haar_idwt(haar_dwt(x)) == x EXACTLY over Q");
                  }
              })
        .test("haar_odd_length_domain_error",
              [](TestContext& t) {
                  auto x = rats({1, 2, 3});
                  auto fwd = wl::haar_dwt(x);
                  t.expect(!fwd.has_value() && fwd.error() == MathError::domain_error,
                           "odd length -> domain_error");
              })
        .test("haar_constant_signal_zero_details",
              [](TestContext& t) {
                  // A constant signal has all-zero detail and a constant (doubled) approx.
                  auto x = rats({5, 5, 5, 5, 5, 5, 5, 5});
                  auto fwd = wl::haar_dwt(x);
                  t.expect(fwd.has_value(), "haar_dwt on constant signal");
                  if (fwd) {
                      bool all_zero = true;
                      for (const Rational& d : fwd->detail) {
                          all_zero = all_zero && (d == Rational::from_int(0));
                      }
                      t.expect(all_zero, "all detail coefficients are exactly 0");
                      bool approx_const = true;
                      for (const Rational& a : fwd->approx) {
                          approx_const = approx_const && (a == Rational::from_int(10));
                      }
                      t.expect(approx_const, "approx coefficients are exactly 2*c == 10");
                  }
                  // Full multi-level decomposition collapses to a single approx coeff,
                  // with every detail band identically zero.
                  auto md = wl::haar_dwt_multi(x, 3);
                  t.expect(md.has_value(), "3-level Haar of length-8 constant signal");
                  if (md) {
                      t.expect(md->approx.size() == 1, "final approximation is a single coeff");
                      bool details_zero = true;
                      for (const auto& band : md->details) {
                          for (const Rational& d : band) {
                              details_zero = details_zero && (d == Rational::from_int(0));
                          }
                      }
                      t.expect(details_zero, "every multi-level detail band is exactly 0");
                  }
              })
        .test("haar_multilevel_roundtrip_exact",
              [](TestContext& t) {
                  auto x = rats({3, -1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3});
                  auto md = wl::haar_dwt_multi(x, 4);
                  t.expect(md.has_value(), "4-level Haar decomposition of length-16");
                  if (md) {
                      auto rec = wl::haar_idwt_multi(*md);
                      t.expect(rec.has_value() && equal_exact(*rec, x),
                               "multi-level Haar round-trip is EXACT over Q");
                  }
              })
        .test("haar_lifting_equals_filterbank_exact",
              [](TestContext& t) {
                  auto x = rats({3, -1, 4, 1, 5, 9, 2, 6});
                  auto fb = wl::haar_dwt(x);
                  auto lift = wl::haar_lifting_forward(x);
                  t.expect(fb.has_value() && lift.has_value(), "both Haar paths succeed");
                  if (fb && lift) {
                      t.expect(equal_exact(fb->approx, lift->approx),
                               "lifting approx == filter-bank approx EXACTLY");
                      t.expect(equal_exact(fb->detail, lift->detail),
                               "lifting detail == filter-bank detail EXACTLY");
                      auto rec = wl::haar_lifting_inverse(lift->approx, lift->detail);
                      t.expect(rec.has_value() && equal_exact(*rec, x),
                               "lifting inverse reconstructs EXACTLY over Q");
                  }
              })
        .test("daubechies_db2_sum_rules_and_orthogonality",
              [](TestContext& t) {
                  auto fb = wl::daubechies(2);
                  t.expect(fb.has_value(), "db2 filter bank available");
                  if (fb) {
                      // Sum rule: Sum h == sqrt(2).
                      t.expect(close(wl::lowpass_sum(fb->analysis_lo), std::numbers::sqrt2),
                               "db2 low-pass sums to sqrt(2)");
                      // db2 has 2 vanishing moments (m = 0, 1).
                      for (int m = 0; m < 2; ++m) {
                          t.expect(close(wl::lowpass_alternating_moment(fb->analysis_lo, m), 0.0,
                                         1e-8),
                                   "db2 vanishing moment (alternating low-pass) == 0");
                          t.expect(close(wl::highpass_moment(fb->analysis_hi, m), 0.0, 1e-8),
                                   "db2 high-pass moment == 0");
                      }
                      // Orthonormality: Sum h^2 == 1 and Sum h_k h_{k+2} == 0.
                      t.expect(close(wl::orthogonality_defect(fb->analysis_lo, 0), 0.0, 1e-10),
                               "db2 Sum h^2 == 1");
                      t.expect(close(wl::orthogonality_defect(fb->analysis_lo, 1), 0.0, 1e-10),
                               "db2 double-shift autocorrelation == 0");
                  }
              })
        .test("daubechies_db4_sum_rules_and_orthogonality",
              [](TestContext& t) {
                  auto fb = wl::daubechies(4);
                  t.expect(fb.has_value(), "db4 filter bank available");
                  if (fb) {
                      t.expect(close(wl::lowpass_sum(fb->analysis_lo), std::numbers::sqrt2),
                               "db4 low-pass sums to sqrt(2)");
                      // db4 has 4 vanishing moments (m = 0..3).
                      for (int m = 0; m < 4; ++m) {
                          t.expect(close(wl::highpass_moment(fb->analysis_hi, m), 0.0, 1e-7),
                                   "db4 high-pass moment == 0");
                      }
                      // Orthonormality across all admissible even shifts (table-sourced
                      // constants: tolerance reflects tabulation precision, not exactness).
                      t.expect(close(wl::orthogonality_defect(fb->analysis_lo, 0), 0.0, 1e-9),
                               "db4 Sum h^2 == 1");
                      for (int m = 1; m <= 3; ++m) {
                          t.expect(close(wl::orthogonality_defect(fb->analysis_lo, m), 0.0, 1e-9),
                                   "db4 double-shift autocorrelation == 0");
                      }
                  }
              })
        .test("daubechies_numeric_roundtrip",
              [](TestContext& t) {
                  auto fb = wl::daubechies(4);
                  std::vector<double> x{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0};
                  t.expect(fb.has_value(), "db4 available");
                  if (fb) {
                      auto lvl = wl::dwt(x, *fb);
                      t.expect(lvl.has_value(), "db4 dwt succeeds");
                      if (lvl) {
                          auto rec = wl::idwt(lvl->approx, lvl->detail, *fb);
                          t.expect(rec.has_value(), "db4 idwt succeeds");
                          bool ok = rec.has_value() && rec->size() == x.size();
                          for (std::size_t i = 0; ok && i < x.size(); ++i) {
                              ok = ok && close((*rec)[i], x[i], 1e-9);
                          }
                          t.expect(ok, "db4 DWT/IDWT reconstructs to round-off (NUMERICAL)");
                      }
                  }
              })
        .test("biorthogonal_cdf53_perfect_reconstruction",
              [](TestContext& t) {
                  std::vector<double> x{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0,
                                        0.25, -0.75, 6.0, -2.0, 1.0, 1.0, -4.0, 2.0};
                  auto lvl = wl::cdf53_forward(x);
                  t.expect(lvl.has_value(), "CDF 5/3 forward succeeds");
                  if (lvl) {
                      auto rec = wl::cdf53_inverse(lvl->approx, lvl->detail);
                      t.expect(rec.has_value(), "CDF 5/3 inverse succeeds");
                      bool ok = rec.has_value() && rec->size() == x.size();
                      for (std::size_t i = 0; ok && i < x.size(); ++i) {
                          ok = ok && close((*rec)[i], x[i], 1e-10);
                      }
                      t.expect(ok, "biorthogonal (5/3 lifting) perfect reconstruction");
                  }
                  // The tabulated filter pair should exist and obey the sqrt(2) sum rule.
                  auto pair = wl::biorthogonal_filters(2, 2);
                  t.expect(pair.has_value(), "bior2.2 tabulated filter pair available");
                  if (pair) {
                      t.expect(close(wl::lowpass_sum(pair->analysis_lo), std::numbers::sqrt2, 1e-9),
                               "bior2.2 analysis low-pass sums to sqrt(2)");
                  }
              })
        .test("swt_length_invariance",
              [](TestContext& t) {
                  auto fb = wl::daubechies(2);
                  std::vector<double> x{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0};
                  t.expect(fb.has_value(), "db2 available");
                  if (fb) {
                      auto s1 = wl::swt(x, *fb, 1);
                      t.expect(s1.has_value(), "SWT level 1 succeeds");
                      t.expect(s1.has_value() && s1->approx.size() == x.size() &&
                                   s1->detail.size() == x.size(),
                               "SWT level 1 preserves length (undecimated)");
                      auto s2 = wl::swt(x, *fb, 2);
                      t.expect(s2.has_value() && s2->approx.size() == x.size() &&
                                   s2->detail.size() == x.size(),
                               "SWT level 2 (a-trous dilated) also preserves length");
                  }
              })
        .test("wavelet_packet_leaf_count_and_sizes",
              [](TestContext& t) {
                  auto fb = wl::daubechies(2);
                  std::vector<double> x{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0};
                  if (fb) {
                      auto wp = wl::wavelet_packet(x, *fb, 2);
                      t.expect(wp.has_value(), "wavelet packet level 2 succeeds");
                      if (wp) {
                          t.expect(wp->size() == 4, "level-2 packet tree has 4 leaves");
                          bool sizes = true;
                          for (const auto& band : *wp) {
                              sizes = sizes && (band.size() == x.size() / 4);
                          }
                          t.expect(sizes, "each leaf band has length n / 2^level");
                      }
                  }
              })
        .test("cwt_peak_aligns_with_feature",
              [](TestContext& t) {
                  // A single localized feature (spike) at index p; the Ricker CWT magnitude
                  // must peak at translation b == p (arg == 0 maximizes |ricker|).
                  const std::size_t n = 32;
                  const std::size_t p = 12;
                  std::vector<double> x(n, 0.0);
                  x[p] = 1.0;
                  std::vector<double> scales{1.0, 2.0, 4.0};
                  auto coeffs = wl::cwt(
                      x, scales, [](double tt) { return std::complex<double>{wl::ricker(tt), 0.0}; });
                  t.expect(coeffs.has_value(), "cwt succeeds");
                  if (coeffs) {
                      double best = -1.0;
                      std::size_t best_b = 0;
                      for (const auto& row : *coeffs) {
                          for (std::size_t b = 0; b < row.size(); ++b) {
                              const double mag = std::abs(row[b]);
                              if (mag > best) {
                                  best = mag;
                                  best_b = b;
                              }
                          }
                      }
                      t.expect(best_b == p, "CWT magnitude peaks at the feature location");
                  }
              })
        .test("cwt_nonpositive_scale_domain_error",
              [](TestContext& t) {
                  std::vector<double> x{1.0, 2.0, 3.0, 4.0};
                  std::vector<double> scales{1.0, 0.0};
                  auto coeffs = wl::cwt(
                      x, scales, [](double tt) { return std::complex<double>{wl::ricker(tt), 0.0}; });
                  t.expect(!coeffs.has_value() && coeffs.error() == MathError::domain_error,
                           "a non-positive scale -> domain_error");
              })
        .test("curvelet_shearlet_honestly_not_implemented",
              [](TestContext& t) {
                  std::vector<double> img(16, 0.0);
                  auto c = wl::curvelet_transform_2d(img, 4, 4);
                  auto s = wl::shearlet_transform_2d(img, 4, 4);
                  t.expect(!c.has_value() && c.error() == MathError::not_implemented,
                           "curvelet transform is honestly not_implemented");
                  t.expect(!s.has_value() && s.error() == MathError::not_implemented,
                           "shearlet transform is honestly not_implemented");
              })
        .run();
}
