// Tests for nimblecas.bigeigen: exact characteristic polynomials and rational eigenvalues
// over BigRational (Faddeev-LeVerrier + rational-root deflation), including the honesty
// boundary (irrational / complex eigenvalues are NOT extracted) and a large-entry case that
// overflows int64 but stays exact in the big tier.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigmatrix;
import nimblecas.bigeigen;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BigMatrix;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// The integer n lifted into Q as n/1.
[[nodiscard]] auto bi(std::int64_t v) -> BigRational {
    return BigRational::from_int(v);
}

// Exact BigRational fraction n/d (denominator known non-zero at each call site).
[[nodiscard]] auto brat(std::int64_t n, std::int64_t d) -> BigRational {
    return BigRational::make(BigInt::from_i64(n), BigInt::from_i64(d)).value();
}

// A BigRational parsed from a decimal string (for magnitudes beyond int64).
[[nodiscard]] auto brs(std::string_view s) -> BigRational {
    return BigRational::from_string(s).value();
}

// Build a BigMatrix from integer rows (low-index row first).
[[nodiscard]] auto bmat(std::vector<std::vector<std::int64_t>> rows) -> BigMatrix {
    std::vector<std::vector<BigRational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<BigRational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(BigRational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return BigMatrix::from_rows(std::move(r)).value();
}

// A dense polynomial equality helper: does the returned coefficient vector match `expected`
// (both low-degree-first)?
[[nodiscard]] auto poly_eq(const std::vector<BigRational>& got,
                           const std::vector<BigRational>& expected) -> bool {
    return got == expected;
}

// Look up the multiplicity recorded for eigenvalue `value` (0 if absent).
[[nodiscard]] auto mult_of(const std::vector<std::pair<BigRational, std::int64_t>>& eig,
                           const BigRational& value) -> std::int64_t {
    for (const auto& [v, m] : eig) {
        if (v == value) {
            return m;
        }
    }
    return 0;
}

// Total number of rational eigenvalues counted with multiplicity.
[[nodiscard]] auto total_mult(const std::vector<std::pair<BigRational, std::int64_t>>& eig)
    -> std::int64_t {
    std::int64_t s = 0;
    for (const auto& [v, m] : eig) {
        s += m;
    }
    return s;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigeigen")
        .test("charpoly_diagonal_2_3_5",
              [](TestContext& t) {
                  // det(xI - diag(2,3,5)) = (x-2)(x-3)(x-5) = x^3 - 10x^2 + 31x - 30.
                  // Low-degree-first: [-30, 31, -10, 1].
                  auto p = nimblecas::characteristic_polynomial(bmat({{2, 0, 0},
                                                                      {0, 3, 0},
                                                                      {0, 0, 5}}))
                               .value();
                  t.expect(poly_eq(p, {bi(-30), bi(31), bi(-10), bi(1)}),
                           "charpoly diag(2,3,5) = x^3 -10x^2 +31x -30");
                  t.expect(p.size() == 4 && p.back() == bi(1), "monic, degree 3");
              })
        .test("charpoly_2x2_hand_check",
              [](TestContext& t) {
                  // [[1,2],[3,4]]: trace 5, det 1*4-2*3 = -2 -> x^2 - 5x - 2.
                  auto p = nimblecas::characteristic_polynomial(bmat({{1, 2}, {3, 4}})).value();
                  t.expect(poly_eq(p, {bi(-2), bi(-5), bi(1)}), "charpoly = x^2 -5x -2");

                  // 1x1: [[7]] -> x - 7.
                  auto p1 = nimblecas::characteristic_polynomial(bmat({{7}})).value();
                  t.expect(poly_eq(p1, {bi(-7), bi(1)}), "charpoly [[7]] = x - 7");

                  // 0x0: the empty-product characteristic polynomial is the constant 1.
                  auto p0 = nimblecas::characteristic_polynomial(BigMatrix::identity(0)).value();
                  t.expect(poly_eq(p0, {bi(1)}), "charpoly of 0x0 = 1");
              })
        .test("charpoly_companion_matrix",
              [](TestContext& t) {
                  // Companion matrix of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6.
                  // Bottom-companion form has that polynomial as its characteristic polynomial.
                  auto c = bmat({{0, 0, 6}, {1, 0, -11}, {0, 1, 6}});
                  auto p = nimblecas::characteristic_polynomial(c).value();
                  t.expect(poly_eq(p, {bi(-6), bi(11), bi(-6), bi(1)}),
                           "companion charpoly = x^3 -6x^2 +11x -6");
                  // ... and its rational eigenvalues are exactly {1, 2, 3}.
                  auto eig = nimblecas::rational_eigenvalues(c).value();
                  t.expect(eig.size() == 3 && total_mult(eig) == 3, "3 simple rational eigenvalues");
                  t.expect(mult_of(eig, bi(1)) == 1 && mult_of(eig, bi(2)) == 1 &&
                               mult_of(eig, bi(3)) == 1,
                           "eigenvalues {1,2,3} each multiplicity 1");
              })
        .test("charpoly_rational_entries",
              [](TestContext& t) {
                  // [[1/2, 0],[0, 1/3]]: (x - 1/2)(x - 1/3) = x^2 - 5/6 x + 1/6.
                  auto m = BigMatrix::from_rows({{brat(1, 2), bi(0)}, {bi(0), brat(1, 3)}}).value();
                  auto p = nimblecas::characteristic_polynomial(m).value();
                  t.expect(poly_eq(p, {brat(1, 6), brat(-5, 6), bi(1)}),
                           "charpoly = x^2 -5/6 x +1/6 (exact fractions)");
                  // Rational eigenvalues 1/2 and 1/3 (denominators divide the cleared leading coeff).
                  auto eig = nimblecas::rational_eigenvalues(m).value();
                  t.expect(mult_of(eig, brat(1, 2)) == 1 && mult_of(eig, brat(1, 3)) == 1,
                           "eigenvalues {1/2, 1/3}");
                  t.expect(total_mult(eig) == 2, "fully rational spectrum");
              })
        .test("rational_eigenvalues_diag_multiplicities",
              [](TestContext& t) {
                  auto eig = nimblecas::rational_eigenvalues(bmat({{2, 0, 0},
                                                                   {0, 3, 0},
                                                                   {0, 0, 5}}))
                                 .value();
                  t.expect(eig.size() == 3, "three distinct eigenvalues");
                  t.expect(mult_of(eig, bi(2)) == 1 && mult_of(eig, bi(3)) == 1 &&
                               mult_of(eig, bi(5)) == 1,
                           "{2,3,5} each simple");
                  // Sorted ascending by the module contract.
                  t.expect(eig.front().first == bi(2) && eig.back().first == bi(5),
                           "eigenvalues sorted ascending");
              })
        .test("rational_eigenvalues_repeated_root",
              [](TestContext& t) {
                  // Jordan block [[2,1],[0,2]] -> (x-2)^2: a single eigenvalue 2 of multiplicity 2.
                  auto eig = nimblecas::rational_eigenvalues(bmat({{2, 1}, {0, 2}})).value();
                  t.expect(eig.size() == 1, "one distinct eigenvalue");
                  t.expect(mult_of(eig, bi(2)) == 2, "eigenvalue 2 with multiplicity 2");
                  t.expect(total_mult(eig) == 2, "multiplicities sum to n = 2");

                  // A 3x3 with a triple root: (x-4)^3 via an upper-triangular block.
                  auto eig3 =
                      nimblecas::rational_eigenvalues(bmat({{4, 1, 5}, {0, 4, 1}, {0, 0, 4}}))
                          .value();
                  t.expect(eig3.size() == 1 && mult_of(eig3, bi(4)) == 3,
                           "triangular (x-4)^3 -> eigenvalue 4 multiplicity 3");
              })
        .test("honesty_complex_eigenvalues_none_rational",
              [](TestContext& t) {
                  // Rotation [[0,1],[-1,0]] has eigenvalues +/- i: charpoly x^2 + 1, NO rational
                  // roots. rational_eigenvalues must (honestly) return an empty list.
                  auto rot = bmat({{0, 1}, {-1, 0}});
                  auto p = nimblecas::characteristic_polynomial(rot).value();
                  t.expect(poly_eq(p, {bi(1), bi(0), bi(1)}), "charpoly = x^2 + 1");
                  auto eig = nimblecas::rational_eigenvalues(rot).value();
                  t.expect(eig.empty(), "no rational eigenvalues (spectrum is complex)");
                  t.expect(total_mult(eig) == 0, "0 < n = 2: spectrum NOT fully rational");
              })
        .test("honesty_irrational_surd_eigenvalues",
              [](TestContext& t) {
                  // [[0,1],[2,0]] has eigenvalues +/- sqrt(2): charpoly x^2 - 2, no rational roots.
                  auto m = bmat({{0, 1}, {2, 0}});
                  auto p = nimblecas::characteristic_polynomial(m).value();
                  t.expect(poly_eq(p, {bi(-2), bi(0), bi(1)}), "charpoly = x^2 - 2");
                  auto eig = nimblecas::rational_eigenvalues(m).value();
                  t.expect(eig.empty(), "no rational eigenvalues (surds omitted)");

                  // Mixed: [[1,2],[3,4]] has the irrational (5 +/- sqrt(33))/2 -> also empty.
                  auto eig2 = nimblecas::rational_eigenvalues(bmat({{1, 2}, {3, 4}})).value();
                  t.expect(eig2.empty(), "irrational eigenvalues omitted");
              })
        .test("rational_eigenvalues_symmetric_pm_one",
              [](TestContext& t) {
                  // [[0,1],[1,0]] (a reflection) has rational eigenvalues +/- 1: charpoly x^2 - 1.
                  auto m = bmat({{0, 1}, {1, 0}});
                  auto eig = nimblecas::rational_eigenvalues(m).value();
                  t.expect(eig.size() == 2, "two eigenvalues");
                  t.expect(mult_of(eig, bi(1)) == 1 && mult_of(eig, bi(-1)) == 1, "{-1, +1}");
                  t.expect(eig.front().first == bi(-1), "ascending: -1 first");
              })
        .test("rational_eigenvalue_zero",
              [](TestContext& t) {
                  // A singular matrix with a zero eigenvalue: [[0,0],[0,5]] -> {0, 5}.
                  auto eig = nimblecas::rational_eigenvalues(bmat({{0, 0}, {0, 5}})).value();
                  t.expect(mult_of(eig, bi(0)) == 1 && mult_of(eig, bi(5)) == 1,
                           "zero is found as an eigenvalue");
              })
        .test("large_entries_overflow_int64_stay_exact",
              [](TestContext& t) {
                  // A 2x2 with trace 8 and det 15 (eigenvalues 3 and 5), but built with an
                  // entry near 10^10 and an off-diagonal near -10^20. The Faddeev-LeVerrier
                  // intermediate products (e.g. 10^10 * (10^10 - 8) ~ 10^20) blow past the
                  // int64 ceiling, yet BigRational keeps everything exact.
                  //   a = 10^10, d = 8 - 10^10, c = 1, b = a*d - 15 = -99999999920000000015.
                  auto big = BigMatrix::from_rows(
                                 {{brs("10000000000"), brs("-99999999920000000015")},
                                  {bi(1), brs("-9999999992")}})
                                 .value();
                  auto p = nimblecas::characteristic_polynomial(big).value();
                  // charpoly = x^2 - 8x + 15 = (x-3)(x-5).
                  t.expect(poly_eq(p, {bi(15), bi(-8), bi(1)}),
                           "large-entry charpoly = x^2 -8x +15 (exact)");
                  auto eig = nimblecas::rational_eigenvalues(big).value();
                  t.expect(mult_of(eig, bi(3)) == 1 && mult_of(eig, bi(5)) == 1,
                           "exact rational eigenvalues {3,5} despite int64-overflowing entries");
                  t.expect(total_mult(eig) == 2, "fully rational spectrum recovered");
              })
        .test("determinant_byproduct_matches_bareiss",
              [](TestContext& t) {
                  // The Faddeev-LeVerrier determinant byproduct agrees with BigMatrix's
                  // fraction-free Bareiss determinant.
                  auto a = bmat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
                  t.expect(nimblecas::determinant(a).value() == a.determinant().value(),
                           "FL det == Bareiss det (3x3)");
                  t.expect(nimblecas::determinant(a).value() == bi(-306), "value is -306");

                  // diag(2,3,5): det = 30.
                  t.expect(nimblecas::determinant(bmat({{2, 0, 0}, {0, 3, 0}, {0, 0, 5}}))
                                   .value() == bi(30),
                           "det diag(2,3,5) = 30");

                  // 0x0 determinant is the empty product 1.
                  t.expect(nimblecas::determinant(BigMatrix::identity(0)).value() == bi(1),
                           "det of 0x0 = 1");
              })
        .test("inverse_byproduct_and_singular",
              [](TestContext& t) {
                  // A * A^{-1} == I for an invertible matrix.
                  auto a = bmat({{1, 2}, {3, 4}});  // det -2, invertible
                  auto inv = nimblecas::inverse(a).value();
                  t.expect(a.multiply(inv).value() == BigMatrix::identity(2),
                           "A * inverse(A) == I");
                  // Exact fractional inverse entries: [[-2, 1],[3/2, -1/2]].
                  t.expect(inv.at(0, 0) == bi(-2) && inv.at(0, 1) == bi(1) &&
                               inv.at(1, 0) == brat(3, 2) && inv.at(1, 1) == brat(-1, 2),
                           "inverse entries exact");

                  // diag(2,3,5) inverts to diag(1/2,1/3,1/5).
                  auto d = bmat({{2, 0, 0}, {0, 3, 0}, {0, 0, 5}});
                  auto dinv = nimblecas::inverse(d).value();
                  t.expect(dinv.at(0, 0) == brat(1, 2) && dinv.at(1, 1) == brat(1, 3) &&
                               dinv.at(2, 2) == brat(1, 5),
                           "diagonal inverse = diag(1/2,1/3,1/5)");

                  // A singular matrix has no inverse -> domain_error.
                  auto singular = bmat({{1, 2}, {2, 4}});  // det 0
                  t.expect(nimblecas::inverse(singular).error() == MathError::domain_error,
                           "singular inverse -> domain_error");
              })
        .test("non_square_domain_errors",
              [](TestContext& t) {
                  auto wide = bmat({{1, 2, 3}, {4, 5, 6}});
                  t.expect(nimblecas::characteristic_polynomial(wide).error() ==
                               MathError::domain_error,
                           "non-square charpoly -> domain_error");
                  t.expect(nimblecas::determinant(wide).error() == MathError::domain_error,
                           "non-square determinant -> domain_error");
                  t.expect(nimblecas::inverse(wide).error() == MathError::domain_error,
                           "non-square inverse -> domain_error");
                  t.expect(nimblecas::rational_eigenvalues(wide).error() ==
                               MathError::domain_error,
                           "non-square rational_eigenvalues -> domain_error");
              })
        .test("honesty_mixed_spectrum_partial_detection",
              [](TestContext& t) {
                  // block-diag([1], [[0,1],[2,0]]) has spectrum {1, +sqrt2, -sqrt2}: ONE
                  // rational eigenvalue and two surds. This is the interesting honesty case
                  // 0 < total_mult < n -- the rational part is found, the surds omitted, and
                  // the caller can tell the spectrum is only PARTLY rational.
                  auto m = bmat({{1, 0, 0}, {0, 0, 1}, {0, 2, 0}});
                  auto p = nimblecas::characteristic_polynomial(m).value();
                  // (x-1)(x^2-2) = x^3 - x^2 - 2x + 2.
                  t.expect(poly_eq(p, {bi(2), bi(-2), bi(-1), bi(1)}),
                           "charpoly = x^3 -x^2 -2x +2");
                  auto eig = nimblecas::rational_eigenvalues(m).value();
                  t.expect(mult_of(eig, bi(1)) == 1, "the one rational eigenvalue 1 is found");
                  t.expect(total_mult(eig) == 1 && total_mult(eig) < 3,
                           "0 < total_mult(1) < n(3): spectrum only partly rational");
              })
        .test("rational_eigenvalue_negative_fraction",
              [](TestContext& t) {
                  // diag(-3/2, 1/2): a NEGATIVE fractional eigenvalue exercises the
                  // pos.negate() branch on a non-integer candidate p/q.
                  auto m = BigMatrix::from_rows({{brat(-3, 2), bi(0)}, {bi(0), brat(1, 2)}}).value();
                  auto p = nimblecas::characteristic_polynomial(m).value();
                  // (x+3/2)(x-1/2) = x^2 + x - 3/4.
                  t.expect(poly_eq(p, {brat(-3, 4), bi(1), bi(1)}), "charpoly = x^2 + x - 3/4");
                  auto eig = nimblecas::rational_eigenvalues(m).value();
                  t.expect(mult_of(eig, brat(-3, 2)) == 1 && mult_of(eig, brat(1, 2)) == 1,
                           "eigenvalues {-3/2, 1/2}");
                  t.expect(total_mult(eig) == 2 && eig.front().first == brat(-3, 2),
                           "fully rational, ascending: -3/2 first");
              })
        .run();
}
