// NimbleCAS Python bindings via nanobind (ROADMAP 10).
// @author Olumuyiwa Oluwasanmi
//
// This is a non-module translation unit: it #includes the nanobind headers (which
// pull in the standard library classically) and `import`s the NimbleCAS C++23
// modules. The std::string / std::vector types from the headers and from the
// imported modules are the same entities, so they interoperate directly.
//
// The C++ core is exception-free (Rule 32, std::expected). At the Python boundary a
// MathError is translated into a Python exception (nb::value_error), which is the
// idiomatic Python contract. A single templated `unwrap` performs that translation
// for every Result<T> (Expr, Rational, Matrix, vectors, pairs, ...).
//
// Surface exposed (see tests/test_bindings.py for a worked example of each):
//   Value types : Expr, Rational, RationalPoly, Matrix, Complex, ComplexMatrix, Laurent.
//   Small POD   : Root, TestStatistic, PeriodicCF, MleModel, ExactOrthogonalQr,
//                 NumericQr, NumericSchur, SeriesCF (read-only fields).
//   Symbolic    : free_of, substitute, simplify, differentiate, polynomial_gcd,
//                 square_free_factor, expand, integrate, integrate_definite.
//   Polynomial  : solve_poly, factor_over_Q, companion_eigenvalues, eigenvalues_qr.
//   Linear alg. : kronecker_product / _sum, direct_sum, hadamard_product, vec, unvec,
//                 exact_orthogonal_qr, numeric_qr, real_schur, schur_eigenvalues,
//                 hermitian_eigenvalues, complex_eigenvalues.
//   Number thy. : from_rational, convergents, reconstruct, quadratic_irrational_cf,
//                 viskovatov (SeriesCF), and the Laurent series (from_rational_function,
//                 valuation, residue, principal/regular part).
//   Canonical   : invariant_factors, minimal_polynomial, companion_matrix,
//                 rational_canonical_form (Frobenius form, exact over Q).
//   Dynamics    : is_perfect_square, classify_phase_portrait (PhasePortrait + PhaseType /
//                 Stability enums), classify_linear_stability (StabilityClassification +
//                 LinearStability enum), fixed_point_affine, is_asymptotically_stable,
//                 classify_equilibrium.
//   Alg. number : NumberField (Q[x]/(m)) and AlgebraicNumber (exact arithmetic, norm/trace)
//                 for a simple algebraic extension Q(alpha).
//   Jordan      : rational_jordan_form (RationalJordan over Q) and jordan_form
//                 (AlgebraicJordan over a quadratic extension Q(alpha)).
//   Statistics  : mean, variance, median, covariance, covariance_matrix, weighted_mean,
//                 raw_moment, central_moment, mode, modes, quantile, data_range, iqr,
//                 skewness_squared, excess_kurtosis, pearson_correlation_squared,
//                 correlation_squared_matrix, coefficient_of_variation_squared; the
//                 hypothesis-test family
//                 (one/two-sample & paired t^2, z^2, chi-squared GoF / independence,
//                 variance-ratio F, one-way ANOVA F, exceeds), Wald / score statistics,
//                 and the Bernoulli/Poisson/Exponential/Normal/Geometric MLE (both the
//                 symbolic model and the exact point estimate from data).
//   Prob. dist. : probdist submodule — an exact symbolic distribution catalog (DistInfo
//                 with mgf/pgf/mean/variance) plus raw_moment / cumulant / factorial_moment,
//                 characteristic_function / laplace_stieltjes, and the tail / concentration
//                 bounds (Markov, Chebyshev, Cantelli, Chernoff, Hoeffding, Bernstein,
//                 McDiarmid, Azuma).
//   GPU         : gpu submodule (present iff HAS_GPU) — device_count, available, and the
//                 poly_eval / edit_distance_batch / bfs / nqueens_count /
//                 qmc_poly_integrate / haar_dwt_batch / batched_matmul / fft_batch kernels.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/complex.h>

import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.polyexpr;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.complex;
import nimblecas.cmatrix;
import nimblecas.solve;
import nimblecas.factor;
import nimblecas.expand;
import nimblecas.symint;
import nimblecas.numeigen;
import nimblecas.qrschur;
import nimblecas.cheigen;
import nimblecas.laurent;
import nimblecas.stats;
import nimblecas.probdist;
import nimblecas.kronprod;
import nimblecas.contfrac;
import nimblecas.hyptest;
import nimblecas.frobenius;
import nimblecas.dynamics;
import nimblecas.algnum;
import nimblecas.jordan;
// The GPU module is bound only when the binding TU is compiled with
// -DNIMBLECAS_EXT_WITH_GPU (and nimblecas_ext is linked against nimblecas_gpu +
// the CUDA runtime). This keeps the default CPU-only Python build importable.
#ifdef NIMBLECAS_EXT_WITH_GPU
import nimblecas.gpu;
#endif

namespace nb = nanobind;
using nimblecas::AlgebraicJordan;
using nimblecas::AlgebraicNumber;
using nimblecas::Complex;
using nimblecas::ComplexMatrix;
using nimblecas::DistInfo;
using nimblecas::ExactOrthogonalQr;
using nimblecas::Expr;
using nimblecas::Laurent;
using nimblecas::LinearStability;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::MleModel;
using nimblecas::NumberField;
using nimblecas::NumericQr;
using nimblecas::NumericSchur;
using nimblecas::PeriodicCF;
using nimblecas::PhasePortrait;
using nimblecas::PhaseType;
using nimblecas::Rational;
using nimblecas::RationalJordan;
using nimblecas::RationalPoly;
using nimblecas::Root;
using nimblecas::SeriesCF;
using nimblecas::Stability;
using nimblecas::StabilityClassification;
using nimblecas::TestStatistic;

namespace {

// Unwraps any Result<T>, raising a Python exception carrying the MathError text when the
// operation failed. This is the one place the exception-free core is bridged to Python's
// exception contract; every fallible binding routes through it.
template <typename T>
[[nodiscard]] auto unwrap(nimblecas::Result<T> result) -> T {
    if (!result.has_value()) {
        throw nb::value_error(nimblecas::to_string_view(result.error()).data());
    }
    return std::move(result).value();
}

// A stable 64-bit hash mixer (boost::hash_combine style), so value types with a
// structural __eq__ also get a structural __hash__ (a == b => hash(a) == hash(b)).
[[nodiscard]] auto hash_combine(std::size_t seed, std::size_t v) -> std::size_t {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

[[nodiscard]] auto hash_rational(const Rational& r) -> std::size_t {
    return hash_combine(std::hash<std::int64_t>{}(r.numerator()),
                        std::hash<std::int64_t>{}(r.denominator()));
}

}  // namespace

NB_MODULE(nimblecas_ext, m) {
    m.doc() =
        "NimbleCAS Python bindings: exact symbolic core (Expr) plus the numeric / linear-"
        "algebra / statistics surface (Rational, RationalPoly, Matrix, Complex, solving, "
        "factoring, integration, Kronecker products, continued fractions, hypothesis tests).";

    // =======================================================================
    // Symbolic core — Expr (unchanged from the original binding).
    // =======================================================================
    nb::class_<Expr>(m, "Expr")
        .def_static("symbol", &Expr::symbol, nb::arg("name"))
        .def_static("integer", &Expr::integer, nb::arg("value"))
        .def_static("real", &Expr::real, nb::arg("value"))
        .def_static(
            "rational",
            [](std::int64_t numerator, std::int64_t denominator) {
                return unwrap(Expr::rational(numerator, denominator));
            },
            nb::arg("numerator"), nb::arg("denominator"))
        .def_static("sum", &Expr::sum, nb::arg("terms"))
        .def_static("product", &Expr::product, nb::arg("factors"))
        .def_static("power", &Expr::power, nb::arg("base"), nb::arg("exponent"))
        .def_static("apply", &Expr::apply, nb::arg("name"), nb::arg("args"))
        .def("add", &Expr::add, nb::arg("other"))
        .def("mul", &Expr::mul, nb::arg("other"))
        .def("pow", &Expr::pow, nb::arg("exponent"))
        .def("is_equivalent_to", &Expr::is_equivalent_to, nb::arg("other"))
        .def("to_string", &Expr::to_string)
        .def("__repr__", &Expr::to_string)
        // Return NotImplemented (not a TypeError) for non-Expr comparands, so
        // `expr == 5` yields False per Python convention instead of raising.
        .def("__eq__",
             [](const Expr& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Expr>(other)) {
                     return nb::cast(a.is_equivalent_to(nb::cast<Expr>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__",
             [](const Expr& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Expr>(other)) {
                     return nb::cast(!a.is_equivalent_to(nb::cast<Expr>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        // Structural hash, consistent with __eq__ (a == b => hash(a) == hash(b)).
        .def("__hash__", [](const Expr& a) { return nimblecas::hash_value(a); })
        .def("__add__", &Expr::add, nb::arg("other"))
        .def("__mul__", &Expr::mul, nb::arg("other"));

    // =======================================================================
    // Rational — exact reduced int64 fraction num/den (den > 0).
    // =======================================================================
    nb::class_<Rational>(m, "Rational")
        .def(nb::init<>())  // 0/1
        .def_static(
            "make",
            [](std::int64_t num, std::int64_t den) { return unwrap(Rational::make(num, den)); },
            nb::arg("num"), nb::arg("den"),
            "Construct num/den in lowest terms; raises on a zero denominator or int64 overflow.")
        .def_static("from_int", &Rational::from_int, nb::arg("value"))
        .def("numerator", &Rational::numerator)
        .def("denominator", &Rational::denominator)
        .def("is_zero", &Rational::is_zero)
        .def("is_integer", &Rational::is_integer)
        .def("add", [](const Rational& a, const Rational& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract", [](const Rational& a, const Rational& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("multiply", [](const Rational& a, const Rational& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("divide", [](const Rational& a, const Rational& b) { return unwrap(a.divide(b)); },
             nb::arg("other"))
        .def("negate", [](const Rational& a) { return unwrap(a.negate()); })
        .def("to_string", &Rational::to_string)
        .def("__repr__",
             [](const Rational& r) { return "Rational(" + r.to_string() + ")"; })
        .def("__add__", [](const Rational& a, const Rational& b) { return unwrap(a.add(b)); })
        .def("__sub__", [](const Rational& a, const Rational& b) { return unwrap(a.subtract(b)); })
        .def("__mul__", [](const Rational& a, const Rational& b) { return unwrap(a.multiply(b)); })
        .def("__truediv__", [](const Rational& a, const Rational& b) { return unwrap(a.divide(b)); })
        .def("__neg__", [](const Rational& a) { return unwrap(a.negate()); })
        .def("__eq__",
             [](const Rational& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Rational>(other)) {
                     return nb::cast(a == nb::cast<Rational>(other));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__",
             [](const Rational& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Rational>(other)) {
                     return nb::cast(!(a == nb::cast<Rational>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__hash__", [](const Rational& r) { return hash_rational(r); });

    // =======================================================================
    // RationalPoly — dense univariate polynomial over Rational.
    // Note: the C++ module has no evaluate() method, so none is bound; use the
    // coefficient accessors for pointwise work. divide() returns a (quotient,
    // remainder) tuple.
    // =======================================================================
    nb::class_<RationalPoly>(m, "RationalPoly")
        .def(nb::init<>())  // the zero polynomial
        .def_static("from_coeffs", &RationalPoly::from_coeffs, nb::arg("coeffs"),
                    "Build from ascending coefficients coeffs[i] = coefficient of x^i "
                    "(trailing zeros are trimmed).")
        .def_static("constant", &RationalPoly::constant, nb::arg("c"))
        .def_static("monomial", &RationalPoly::monomial, nb::arg("coeff"), nb::arg("degree"))
        .def("is_zero", &RationalPoly::is_zero)
        .def("degree", &RationalPoly::degree)
        .def("coefficient", &RationalPoly::coefficient, nb::arg("i"),
             "Coefficient of x^i (0 for i beyond the degree).")
        .def("leading_coefficient", &RationalPoly::leading_coefficient)
        .def("coefficients",
             [](const RationalPoly& p) {
                 const auto span = p.coefficients();
                 return std::vector<Rational>(span.begin(), span.end());
             })
        .def("add", [](const RationalPoly& a, const RationalPoly& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract",
             [](const RationalPoly& a, const RationalPoly& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("scale", [](const RationalPoly& a, const Rational& s) { return unwrap(a.scale(s)); },
             nb::arg("s"))
        .def("multiply",
             [](const RationalPoly& a, const RationalPoly& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("monic", [](const RationalPoly& a) { return unwrap(a.monic()); })
        .def("gcd", [](const RationalPoly& a, const RationalPoly& b) { return unwrap(a.gcd(b)); },
             nb::arg("other"))
        .def("derivative", [](const RationalPoly& a) { return unwrap(a.derivative()); })
        .def(
            "divide",
            [](const RationalPoly& a, const RationalPoly& divisor) {
                auto dm = unwrap(a.divide(divisor));
                return std::make_pair(std::move(dm.quotient), std::move(dm.remainder));
            },
            nb::arg("divisor"),
            "Euclidean division: returns (quotient, remainder) with "
            "self == quotient*divisor + remainder and deg(remainder) < deg(divisor).")
        .def("to_string", [](const RationalPoly& p) { return p.to_string(); })
        .def("to_string", [](const RationalPoly& p, const std::string& var) {
            return p.to_string(var);
        }, nb::arg("var"))
        .def("__repr__",
             [](const RationalPoly& p) { return "RationalPoly(" + p.to_string() + ")"; })
        .def("__eq__",
             [](const RationalPoly& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<RationalPoly>(other)) {
                     return nb::cast(a.is_equal(nb::cast<RationalPoly>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__",
             [](const RationalPoly& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<RationalPoly>(other)) {
                     return nb::cast(!a.is_equal(nb::cast<RationalPoly>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__hash__", [](const RationalPoly& p) {
            std::size_t h = 0;
            for (const Rational& c : p.coefficients()) {
                h = hash_combine(h, hash_rational(c));
            }
            return h;
        });

    // =======================================================================
    // Matrix — dense rows x cols grid of exact Rational entries.
    // =======================================================================
    nb::class_<Matrix>(m, "Matrix")
        .def(nb::init<>())  // the empty 0x0 matrix
        .def_static("from_rows",
                    [](std::vector<std::vector<Rational>> rows) {
                        return unwrap(Matrix::from_rows(std::move(rows)));
                    },
                    nb::arg("rows"),
                    "Build from a list of equal-length rows; raises domain_error if ragged.")
        .def_static("identity", &Matrix::identity, nb::arg("n"))
        .def_static("zero", &Matrix::zero, nb::arg("rows"), nb::arg("cols"))
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("is_square", &Matrix::is_square)
        .def(
            "at",
            [](const Matrix& mat, std::size_t i, std::size_t j) -> Rational {
                if (i >= mat.rows() || j >= mat.cols()) {
                    throw nb::index_error("Matrix index out of range");
                }
                return mat.at(i, j);
            },
            nb::arg("i"), nb::arg("j"))
        .def("add", [](const Matrix& a, const Matrix& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract", [](const Matrix& a, const Matrix& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("scale", [](const Matrix& a, const Rational& s) { return unwrap(a.scale(s)); },
             nb::arg("s"))
        .def("multiply", [](const Matrix& a, const Matrix& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("transpose", [](const Matrix& a) { return unwrap(a.transpose()); })
        .def("trace", [](const Matrix& a) { return unwrap(a.trace()); })
        .def("determinant", [](const Matrix& a) { return unwrap(a.determinant()); })
        .def("solve", [](const Matrix& a, const Matrix& b) { return unwrap(a.solve(b)); },
             nb::arg("b"), "Solve A x = b for a square, nonsingular A.")
        .def("inverse", [](const Matrix& a) { return unwrap(a.inverse()); })
        .def("rank", &Matrix::rank)
        .def("to_string", &Matrix::to_string)
        .def("__repr__", &Matrix::to_string)
        .def("__mul__", [](const Matrix& a, const Matrix& b) { return unwrap(a.multiply(b)); })
        .def("__add__", [](const Matrix& a, const Matrix& b) { return unwrap(a.add(b)); })
        .def("__sub__", [](const Matrix& a, const Matrix& b) { return unwrap(a.subtract(b)); })
        .def("__eq__",
             [](const Matrix& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Matrix>(other)) {
                     return nb::cast(a == nb::cast<Matrix>(other));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__",
             [](const Matrix& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Matrix>(other)) {
                     return nb::cast(!(a == nb::cast<Matrix>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__hash__", [](const Matrix& mat) {
            std::size_t h = hash_combine(mat.rows(), mat.cols());
            for (std::size_t i = 0; i < mat.rows(); ++i) {
                for (std::size_t j = 0; j < mat.cols(); ++j) {
                    h = hash_combine(h, hash_rational(mat.at(i, j)));
                }
            }
            return h;
        });

    // =======================================================================
    // Complex — exact Gaussian rational re + im*i.
    // =======================================================================
    nb::class_<Complex>(m, "Complex")
        .def(nb::init<>())  // 0 + 0i
        .def_static("make", &Complex::make, nb::arg("re"), nb::arg("im"))
        .def_static("from_real", &Complex::from_real, nb::arg("re"))
        .def_static("from_int", &Complex::from_int, nb::arg("value"))
        .def_static("i", &Complex::i)
        .def("real", &Complex::real)
        .def("imag", &Complex::imag)
        .def("is_real", &Complex::is_real)
        .def("is_imaginary", &Complex::is_imaginary)
        .def("is_zero", &Complex::is_zero)
        .def("add", [](const Complex& a, const Complex& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract", [](const Complex& a, const Complex& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("multiply", [](const Complex& a, const Complex& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("divide", [](const Complex& a, const Complex& b) { return unwrap(a.divide(b)); },
             nb::arg("other"))
        .def("negate", [](const Complex& a) { return unwrap(a.negate()); })
        .def("conjugate", [](const Complex& a) { return unwrap(a.conjugate()); })
        .def("reciprocal", [](const Complex& a) { return unwrap(a.reciprocal()); })
        .def("norm_squared", [](const Complex& a) { return unwrap(a.norm_squared()); },
             "Exact |z|^2 = re^2 + im^2 (|z| itself is irrational and not provided).")
        .def("to_string", &Complex::to_string)
        .def("__repr__", &Complex::to_string)
        .def("__add__", [](const Complex& a, const Complex& b) { return unwrap(a.add(b)); })
        .def("__sub__", [](const Complex& a, const Complex& b) { return unwrap(a.subtract(b)); })
        .def("__mul__", [](const Complex& a, const Complex& b) { return unwrap(a.multiply(b)); })
        .def("__truediv__", [](const Complex& a, const Complex& b) { return unwrap(a.divide(b)); })
        .def("__neg__", [](const Complex& a) { return unwrap(a.negate()); })
        .def("__eq__",
             [](const Complex& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Complex>(other)) {
                     return nb::cast(a == nb::cast<Complex>(other));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__",
             [](const Complex& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Complex>(other)) {
                     return nb::cast(!(a == nb::cast<Complex>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__hash__", [](const Complex& c) {
            return hash_combine(hash_rational(c.real()), hash_rational(c.imag()));
        });

    // =======================================================================
    // ComplexMatrix — dense rows x cols grid of exact Gaussian-rational Complex
    // entries (the input type of the complex eigensolver, cheigen).
    // =======================================================================
    nb::class_<ComplexMatrix>(m, "ComplexMatrix")
        .def(nb::init<>())  // the empty 0x0 matrix
        .def_static(
            "from_rows",
            [](std::vector<std::vector<Complex>> rows) {
                return unwrap(ComplexMatrix::from_rows(std::move(rows)));
            },
            nb::arg("rows"),
            "Build from a list of equal-length rows of Complex; raises on a ragged or empty list.")
        .def_static("identity", &ComplexMatrix::identity, nb::arg("n"))
        .def_static("zero", &ComplexMatrix::zero, nb::arg("rows"), nb::arg("cols"))
        .def("rows", &ComplexMatrix::rows)
        .def("cols", &ComplexMatrix::cols)
        .def("is_square", &ComplexMatrix::is_square)
        .def(
            "at",
            [](const ComplexMatrix& mat, std::size_t i, std::size_t j) -> Complex {
                if (i >= mat.rows() || j >= mat.cols()) {
                    throw nb::index_error("ComplexMatrix index out of range");
                }
                return mat.at(i, j);
            },
            nb::arg("i"), nb::arg("j"))
        .def("add", [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract",
             [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("scale", [](const ComplexMatrix& a, const Complex& s) { return unwrap(a.scale(s)); },
             nb::arg("s"))
        .def("multiply",
             [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("transpose", [](const ComplexMatrix& a) { return unwrap(a.transpose()); })
        .def("conjugate", [](const ComplexMatrix& a) { return unwrap(a.conjugate()); })
        .def("adjoint", [](const ComplexMatrix& a) { return unwrap(a.adjoint()); },
             "The conjugate transpose A^H (the map defining Hermitian / unitary / normal).")
        .def("is_hermitian", [](const ComplexMatrix& a) { return unwrap(a.is_hermitian()); })
        .def("is_skew_hermitian",
             [](const ComplexMatrix& a) { return unwrap(a.is_skew_hermitian()); })
        .def("is_unitary", [](const ComplexMatrix& a) { return unwrap(a.is_unitary()); })
        .def("is_normal", [](const ComplexMatrix& a) { return unwrap(a.is_normal()); })
        .def("to_string", &ComplexMatrix::to_string)
        .def("__repr__", &ComplexMatrix::to_string)
        .def("__mul__",
             [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.multiply(b)); })
        .def("__add__",
             [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.add(b)); })
        .def("__sub__",
             [](const ComplexMatrix& a, const ComplexMatrix& b) { return unwrap(a.subtract(b)); })
        .def("__eq__",
             [](const ComplexMatrix& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<ComplexMatrix>(other)) {
                     return nb::cast(a == nb::cast<ComplexMatrix>(other));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__", [](const ComplexMatrix& a, nb::handle other) -> nb::object {
            if (nb::isinstance<ComplexMatrix>(other)) {
                return nb::cast(!(a == nb::cast<ComplexMatrix>(other)));
            }
            return nb::borrow(Py_NotImplemented);
        });

    // =======================================================================
    // Laurent — truncated Laurent series over Q (a finite principal part plus a
    // truncated Taylor tail): coefficients c_k for k in [order_min, truncation).
    // =======================================================================
    nb::class_<Laurent>(m, "Laurent")
        .def_static(
            "from_coeffs",
            [](std::int64_t order_min, std::vector<Rational> coeffs) {
                return unwrap(Laurent::from_coeffs(order_min, std::move(coeffs)));
            },
            nb::arg("order_min"), nb::arg("coeffs"),
            "coeffs[i] is the coefficient of x^(order_min + i); the list must be non-empty.")
        .def_static("zero",
                    [](std::int64_t order_min, std::size_t size) {
                        return unwrap(Laurent::zero(order_min, size));
                    },
                    nb::arg("order_min"), nb::arg("size"))
        .def_static("constant",
                    [](const Rational& c, std::size_t size) {
                        return unwrap(Laurent::constant(c, size));
                    },
                    nb::arg("c"), nb::arg("size"))
        .def_static("one", [](std::size_t size) { return unwrap(Laurent::one(size)); },
                    nb::arg("size"))
        .def_static("monomial",
                    [](const Rational& c, std::int64_t exponent, std::int64_t order_min,
                       std::size_t size) {
                        return unwrap(Laurent::monomial(c, exponent, order_min, size));
                    },
                    nb::arg("c"), nb::arg("exponent"), nb::arg("order_min"), nb::arg("size"))
        .def_static(
            "from_rational_function",
            [](const RationalPoly& num, const RationalPoly& den, const Rational& point,
               std::size_t order) {
                return unwrap(Laurent::from_rational_function(num, den, point, order));
            },
            nb::arg("num"), nb::arg("den"), nb::arg("point"), nb::arg("order"),
            "Laurent-expand num/den about `point` in powers of (x - point), keeping `order` "
            "coefficients; a pole yields a genuine negative principal part.")
        .def("order_min", &Laurent::order_min)
        .def("truncation_order", &Laurent::truncation_order)
        .def("size", &Laurent::size)
        .def("coefficient", &Laurent::coefficient, nb::arg("k"),
             "Coefficient of x^k (a genuine 0 below order_min; a truncation 0 at/above "
             "truncation_order).")
        .def("coefficients",
             [](const Laurent& p) {
                 const auto span = p.coefficients();
                 return std::vector<Rational>(span.begin(), span.end());
             })
        .def("valuation", [](const Laurent& p) { return unwrap(p.valuation()); },
             "Exponent of the first nonzero tracked coefficient; raises when the tracked part "
             "is entirely zero.")
        .def("principal_part", [](const Laurent& p) { return unwrap(p.principal_part()); },
             "The finite principal part (terms of exponent < 0).")
        .def("regular_part", [](const Laurent& p) { return unwrap(p.regular_part()); },
             "The regular (Taylor) part (terms of exponent >= 0).")
        .def("residue", [](const Laurent& p) { return unwrap(p.residue()); },
             "The residue c_{-1}; raises when x^{-1} lies beyond the tracked order.")
        .def("add", [](const Laurent& a, const Laurent& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract", [](const Laurent& a, const Laurent& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("scale", [](const Laurent& a, const Rational& s) { return unwrap(a.scale(s)); },
             nb::arg("s"))
        .def("multiply", [](const Laurent& a, const Laurent& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("inverse", [](const Laurent& a) { return unwrap(a.inverse()); })
        .def("divide", [](const Laurent& a, const Laurent& b) { return unwrap(a.divide(b)); },
             nb::arg("other"))
        .def("to_string", [](const Laurent& p) { return p.to_string(); })
        .def("to_string", [](const Laurent& p, const std::string& var) { return p.to_string(var); },
             nb::arg("var"))
        .def("__repr__", [](const Laurent& p) { return "Laurent(" + p.to_string() + ")"; })
        .def("__add__", [](const Laurent& a, const Laurent& b) { return unwrap(a.add(b)); })
        .def("__sub__", [](const Laurent& a, const Laurent& b) { return unwrap(a.subtract(b)); })
        .def("__mul__", [](const Laurent& a, const Laurent& b) { return unwrap(a.multiply(b)); })
        .def("__truediv__", [](const Laurent& a, const Laurent& b) { return unwrap(a.divide(b)); })
        .def("__eq__",
             [](const Laurent& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<Laurent>(other)) {
                     return nb::cast(a.is_equal(nb::cast<Laurent>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__", [](const Laurent& a, nb::handle other) -> nb::object {
            if (nb::isinstance<Laurent>(other)) {
                return nb::cast(!a.is_equal(nb::cast<Laurent>(other)));
            }
            return nb::borrow(Py_NotImplemented);
        });

    // =======================================================================
    // Small read-only result records.
    // =======================================================================
    // Root: one root of a polynomial (value + exactness flag + optional multiplicity).
    nb::class_<Root>(m, "Root")
        .def_ro("value", &Root::value,
                "The root as a symbolic Expr (a rational, a radical, or a numeric leaf).")
        .def_ro("exact", &Root::exact,
                "True for an exact algebraic root; False for a numeric eigenvalue "
                "approximation (a degree>=5 irreducible factor).")
        .def_ro("multiplicity", &Root::multiplicity,
                "Algebraic multiplicity when known (rational roots), else None.")
        .def("__repr__", [](const Root& r) {
            return "Root(value=" + r.value.to_string() +
                   ", exact=" + (r.exact ? "True" : "False") + ")";
        });

    // TestStatistic: an exact rational statistic with its integer degrees of freedom.
    nb::class_<TestStatistic>(m, "TestStatistic")
        .def_ro("value", &TestStatistic::value)
        .def_ro("df1", &TestStatistic::df1)
        .def_ro("df2", &TestStatistic::df2, "Denominator df (F family), else None.")
        .def("__repr__", [](const TestStatistic& t) {
            return "TestStatistic(value=" + t.value.to_string() +
                   ", df1=" + std::to_string(t.df1) + ")";
        });

    // PeriodicCF: the eventually-periodic continued fraction [prefix; (period)].
    nb::class_<PeriodicCF>(m, "PeriodicCF")
        .def_ro("prefix", &PeriodicCF::prefix)
        .def_ro("period", &PeriodicCF::period);

    // MleModel: the exact symbolic MLE data of a one-parameter family.
    nb::class_<MleModel>(m, "MleModel")
        .def_ro("parameter", &MleModel::parameter)
        .def_ro("log_likelihood", &MleModel::log_likelihood)
        .def_ro("score", &MleModel::score)
        .def_ro("mle", &MleModel::mle)
        .def_ro("fisher_information", &MleModel::fisher_information);

    // ExactOrthogonalQr: the exact-over-Q Gram-Schmidt factorization A = Q*R, with Q's
    // columns mutually orthogonal (not orthonormal) and R upper-triangular unit-diagonal.
    nb::class_<ExactOrthogonalQr>(m, "ExactOrthogonalQr")
        .def_ro("q", &ExactOrthogonalQr::q,
                "m x n Matrix with mutually orthogonal columns (Q^T Q is diagonal); Q*R == A "
                "exactly.")
        .def_ro("r", &ExactOrthogonalQr::r, "n x n upper-triangular Matrix with unit diagonal.");

    // NumericQr: the numeric Householder QR, Q orthonormal, R upper-triangular (row-major
    // double buffers exposed as Python lists).
    nb::class_<NumericQr>(m, "NumericQr")
        .def_ro("rows", &NumericQr::rows)
        .def_ro("cols", &NumericQr::cols)
        .def_ro("q", &NumericQr::q, "m x m orthonormal factor, row-major (length rows*rows).")
        .def_ro("r", &NumericQr::r, "m x n upper-triangular factor, row-major (length rows*cols).");

    // NumericSchur: the numeric real Schur form A = Q*T*Q^T, T quasi-upper-triangular
    // (row-major double buffers exposed as Python lists).
    nb::class_<NumericSchur>(m, "NumericSchur")
        .def_ro("n", &NumericSchur::n)
        .def_ro("q", &NumericSchur::q, "n x n orthogonal Schur vectors, row-major.")
        .def_ro("t", &NumericSchur::t, "n x n quasi-upper-triangular real Schur form, row-major.");

    // SeriesCF: a power series' corresponding continued fraction (C-fraction) b0 + a_1 x /
    // (1 + a_2 x / (1 + ...)), from the Viskovatov algorithm.
    nb::class_<SeriesCF>(m, "SeriesCF")
        .def_ro("b0", &SeriesCF::b0, "The constant term b0 = c_0.")
        .def_ro("a", &SeriesCF::a, "The partial numerators a_1, a_2, ... (stage k is a_k*x/(1+..)).");

    // =======================================================================
    // Symbolic free functions (original + expansion / integration).
    // =======================================================================
    m.def("free_of", &nimblecas::free_of, nb::arg("u"), nb::arg("t"),
          "True if u does not contain the sub-expression t.");
    m.def("substitute", &nimblecas::substitute, nb::arg("u"), nb::arg("t"), nb::arg("r"),
          "Replace every occurrence of t in u with r.");
    m.def(
        "simplify", [](const Expr& u) { return unwrap(nimblecas::simplify(u)); }, nb::arg("u"),
        "Automatically simplify an expression (constant folding, identities, "
        "canonical ordering, like-term combination).");
    m.def(
        "differentiate",
        [](const Expr& u, const std::string& var) {
            return unwrap(nimblecas::differentiate(u, var));
        },
        nb::arg("u"), nb::arg("var"),
        "Differentiate u with respect to the symbol named var (result simplified).");
    m.def(
        "polynomial_gcd",
        [](const Expr& a, const Expr& b, const std::string& var) {
            return unwrap(nimblecas::polynomial_gcd(a, b, var));
        },
        nb::arg("a"), nb::arg("b"), nb::arg("var"),
        "GCD of two univariate polynomial expressions in var.");
    m.def(
        "square_free_factor",
        [](const Expr& u, const std::string& var) {
            return unwrap(nimblecas::square_free_factor(u, var));
        },
        nb::arg("u"), nb::arg("var"),
        "Square-free factorization: list of (factor, multiplicity) for a polynomial in var.");
    m.def(
        "expand", [](const Expr& u) { return unwrap(nimblecas::expand(u)); }, nb::arg("u"),
        "Distribute products over sums and expand integer powers of sums, then simplify.");
    m.def(
        "integrate",
        [](const Expr& f, const std::string& var) { return unwrap(nimblecas::integrate(f, var)); },
        nb::arg("f"), nb::arg("var"),
        "Indefinite integral (no +C); raises 'not implemented' outside the supported class.");
    m.def(
        "integrate_definite",
        [](const Expr& f, const std::string& var, const Expr& a, const Expr& b) {
            return unwrap(nimblecas::integrate_definite(f, var, a, b));
        },
        nb::arg("f"), nb::arg("var"), nb::arg("a"), nb::arg("b"),
        "Definite integral F(b) - F(a) for an antiderivative F.");

    // =======================================================================
    // Polynomial solving / factoring / numeric eigenvalues.
    // =======================================================================
    m.def(
        "solve_poly",
        [](const RationalPoly& p, double tol, std::size_t max_iter) {
            return unwrap(nimblecas::solve_poly(p, tol, max_iter));
        },
        nb::arg("p"), nb::arg("tol") = 1e-12, nb::arg("max_iter") = static_cast<std::size_t>(1000),
        "All roots of p as a list of Root (exact radicals for factors of degree <= 4; "
        "numeric companion-matrix eigenvalues, exact=False, for irreducible degree >= 5).");
    m.def(
        "factor_over_Q",
        [](const RationalPoly& p) { return unwrap(nimblecas::factor_over_Q(p)); }, nb::arg("p"),
        "Factor p into irreducibles over Q: a list of (irreducible RationalPoly, multiplicity).");
    m.def(
        "companion_eigenvalues",
        [](const RationalPoly& p, double tol, std::size_t max_iter) {
            return unwrap(nimblecas::companion_eigenvalues(p, tol, max_iter));
        },
        nb::arg("p"), nb::arg("tol") = 1e-12, nb::arg("max_iter") = static_cast<std::size_t>(1000),
        "Numeric roots of p as the eigenvalues of its companion matrix (list of complex).");
    m.def(
        "eigenvalues_qr",
        [](const std::vector<double>& a, std::size_t n, double tol, std::size_t max_iter) {
            return unwrap(nimblecas::eigenvalues_qr(std::span<const double>(a), n, tol, max_iter));
        },
        nb::arg("a"), nb::arg("n"), nb::arg("tol") = 1e-12,
        nb::arg("max_iter") = static_cast<std::size_t>(1000),
        "All eigenvalues of the n x n real matrix given row-major in `a` (length n*n).");

    // =======================================================================
    // Structured matrix products (kronprod).
    // =======================================================================
    m.def("kronecker_product",
          [](const Matrix& a, const Matrix& b) { return unwrap(nimblecas::kronecker_product(a, b)); },
          nb::arg("a"), nb::arg("b"), "The Kronecker product A (x) B.");
    m.def("kronecker_sum",
          [](const Matrix& a, const Matrix& b) { return unwrap(nimblecas::kronecker_sum(a, b)); },
          nb::arg("a"), nb::arg("b"), "The Kronecker sum A (+) B (square operands).");
    m.def("direct_sum",
          [](const Matrix& a, const Matrix& b) { return unwrap(nimblecas::direct_sum(a, b)); },
          nb::arg("a"), nb::arg("b"), "The block-diagonal direct sum diag(A, B).");
    m.def("hadamard_product",
          [](const Matrix& a, const Matrix& b) { return unwrap(nimblecas::hadamard_product(a, b)); },
          nb::arg("a"), nb::arg("b"), "The entrywise (Hadamard) product A (o) B.");
    m.def("vec", [](const Matrix& a) { return unwrap(nimblecas::vec(a)); }, nb::arg("a"),
          "Column-major vectorisation: stack the columns of A into one column.");
    m.def("unvec",
          [](const Matrix& v, std::size_t rows) { return unwrap(nimblecas::unvec(v, rows)); },
          nb::arg("v"), nb::arg("rows"), "Inverse of vec: reshape a column into a rows x k matrix.");

    // =======================================================================
    // Continued fractions (contfrac).
    // =======================================================================
    m.def("cf_from_rational",
          [](const Rational& r) { return unwrap(nimblecas::from_rational(r)); }, nb::arg("r"),
          "Simple continued fraction [a0; a1, ...] of r as a list of int (Euclidean trace).");
    m.def("cf_convergents",
          [](const std::vector<std::int64_t>& coeffs) {
              return unwrap(nimblecas::convergents(std::span<const std::int64_t>(coeffs)));
          },
          nb::arg("coeffs"), "The rational convergents of a partial-quotient list.");
    m.def("cf_reconstruct",
          [](const std::vector<std::int64_t>& coeffs) {
              return unwrap(nimblecas::reconstruct(std::span<const std::int64_t>(coeffs)));
          },
          nb::arg("coeffs"), "The exact rational value of a finite continued fraction.");
    m.def("quadratic_irrational_cf",
          [](std::int64_t d) { return unwrap(nimblecas::quadratic_irrational_cf(d)); },
          nb::arg("D"),
          "The periodic continued fraction of sqrt(D) for a positive non-square integer D.");
    m.def("viskovatov",
          [](const std::vector<Rational>& series_coeffs) {
              return unwrap(nimblecas::viskovatov(std::span<const Rational>(series_coeffs)));
          },
          nb::arg("series_coeffs"),
          "The corresponding continued fraction (C-fraction, SeriesCF) of a formal power series "
          "c_0 + c_1 x + ...; raises not_implemented when the regular C-fraction does not exist.");

    // =======================================================================
    // Descriptive statistics (stats) — a useful cross-section.
    // =======================================================================
    m.def("mean",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::mean(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The exact arithmetic mean.");
    m.def("variance",
          [](const std::vector<Rational>& data, bool sample) {
              return unwrap(nimblecas::variance(std::span<const Rational>(data), sample));
          },
          nb::arg("data"), nb::arg("sample") = true,
          "The exact variance (sample divisor n-1 when sample=True, else population n).");
    m.def("median",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::median(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The exact median.");
    m.def("covariance",
          [](const std::vector<Rational>& x, const std::vector<Rational>& y, bool sample) {
              return unwrap(nimblecas::covariance(std::span<const Rational>(x),
                                                  std::span<const Rational>(y), sample));
          },
          nb::arg("x"), nb::arg("y"), nb::arg("sample") = true, "The exact covariance of x and y.");
    m.def("weighted_mean",
          [](const std::vector<Rational>& data, const std::vector<Rational>& weights) {
              return unwrap(nimblecas::weighted_mean(std::span<const Rational>(data),
                                                     std::span<const Rational>(weights)));
          },
          nb::arg("data"), nb::arg("weights"),
          "The exact weighted mean (sum w_i x_i)/(sum w_i); raises on a zero total weight.");
    m.def("raw_moment",
          [](const std::vector<Rational>& data, unsigned k) {
              return unwrap(nimblecas::raw_moment(std::span<const Rational>(data), k));
          },
          nb::arg("data"), nb::arg("k"), "The exact k-th raw (about-zero) moment (1/n) sum x_i^k.");
    m.def("central_moment",
          [](const std::vector<Rational>& data, unsigned k) {
              return unwrap(nimblecas::central_moment(std::span<const Rational>(data), k));
          },
          nb::arg("data"), nb::arg("k"),
          "The exact k-th central moment (1/n) sum (x_i - mean)^k (population form).");
    m.def("mode",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::mode(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The most frequent value (ties broken toward the smallest).");
    m.def("quantile",
          [](const std::vector<Rational>& data, const Rational& p) {
              return unwrap(nimblecas::quantile(std::span<const Rational>(data), p));
          },
          nb::arg("data"), nb::arg("p"),
          "The exact type-7 (linear-interpolation) quantile at probability p in [0, 1].");
    m.def("data_range",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::range(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The exact range max - min.");
    m.def("iqr",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::iqr(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The exact interquartile range Q3 - Q1 (type-7 quantiles).");
    m.def("pearson_correlation_squared",
          [](const std::vector<Rational>& x, const std::vector<Rational>& y) {
              return unwrap(nimblecas::pearson_correlation_squared(std::span<const Rational>(x),
                                                                   std::span<const Rational>(y)));
          },
          nb::arg("x"), nb::arg("y"),
          "The exact squared Pearson correlation r^2 = cov^2/(var_x var_y), a rational in [0,1] "
          "(r itself is generally irrational; recover sign(r) = sign(covariance)).");
    m.def("skewness_squared",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::skewness_squared(std::span<const Rational>(data)));
          },
          nb::arg("data"),
          "The exact squared skewness m_3^2 / m_2^3, a rational (sign is sign of central_moment 3).");
    m.def("excess_kurtosis",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::excess_kurtosis(std::span<const Rational>(data)));
          },
          nb::arg("data"), "The exact excess kurtosis m_4 / m_2^2 - 3, a rational.");
    m.def("coefficient_of_variation_squared",
          [](const std::vector<Rational>& data, bool sample) {
              return unwrap(nimblecas::coefficient_of_variation_squared(
                  std::span<const Rational>(data), sample));
          },
          nb::arg("data"), nb::arg("sample") = true,
          "The exact squared coefficient of variation var/mean^2, a rational (cv itself is its "
          "generally-irrational square root).");
    m.def("modes",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::modes(std::span<const Rational>(data)));
          },
          nb::arg("data"),
          "Every value attaining the maximum frequency, ascending (a single-element list when "
          "the mode is unique).");
    m.def("covariance_matrix",
          [](const std::vector<std::vector<Rational>>& variables, bool sample) {
              std::vector<std::span<const Rational>> spans;
              spans.reserve(variables.size());
              for (const auto& v : variables) {
                  spans.emplace_back(v);
              }
              return unwrap(nimblecas::covariance_matrix(spans, sample));
          },
          nb::arg("variables"), nb::arg("sample") = true,
          "The exact symmetric d x d covariance Matrix Sigma_{jk} = covariance(variables[j], "
          "variables[k], sample); every variable must share the common sample length. The "
          "diagonal holds the per-variable variances.");
    m.def("correlation_squared_matrix",
          [](const std::vector<std::vector<Rational>>& variables) {
              std::vector<std::span<const Rational>> spans;
              spans.reserve(variables.size());
              for (const auto& v : variables) {
                  spans.emplace_back(v);
              }
              return unwrap(nimblecas::correlation_squared_matrix(spans));
          },
          nb::arg("variables"),
          "The exact symmetric d x d matrix of squared Pearson correlations "
          "R^2_{jk} = pearson_correlation_squared(variables[j], variables[k]); the diagonal is 1 "
          "(the signed r is omitted, being generally irrational).");

    // =======================================================================
    // Hypothesis tests & maximum-likelihood estimation (hyptest).
    // =======================================================================
    m.def("chi_squared_goodness_of_fit",
          [](const std::vector<Rational>& observed, const std::vector<Rational>& expected) {
              return unwrap(nimblecas::chi_squared_goodness_of_fit(
                  std::span<const Rational>(observed), std::span<const Rational>(expected)));
          },
          nb::arg("observed"), nb::arg("expected"),
          "Exact chi-squared goodness-of-fit statistic sum_i (O_i - E_i)^2 / E_i, df1 = k - 1.");
    m.def("one_sample_t_squared",
          [](const std::vector<Rational>& data, const Rational& mu0) {
              return unwrap(
                  nimblecas::one_sample_t_squared(std::span<const Rational>(data), mu0));
          },
          nb::arg("data"), nb::arg("mu0"),
          "Exact rational t^2 = n (xbar - mu0)^2 / s^2, df1 = n - 1 (t is the irrational square "
          "root; recover sign(t) = sign(xbar - mu0)).");
    m.def("two_sample_t_squared",
          [](const std::vector<Rational>& x, const std::vector<Rational>& y) {
              return unwrap(nimblecas::two_sample_t_squared(std::span<const Rational>(x),
                                                            std::span<const Rational>(y)));
          },
          nb::arg("x"), nb::arg("y"),
          "Exact pooled two-sample t^2, df1 = n1 + n2 - 2.");
    m.def("paired_t_squared",
          [](const std::vector<Rational>& x, const std::vector<Rational>& y) {
              return unwrap(nimblecas::paired_t_squared(std::span<const Rational>(x),
                                                        std::span<const Rational>(y)));
          },
          nb::arg("x"), nb::arg("y"),
          "Exact paired t^2 of the differences x_i - y_i against 0, df1 = n - 1.");
    m.def("z_squared",
          [](const std::vector<Rational>& data, const Rational& mu0, const Rational& pop_variance) {
              return unwrap(
                  nimblecas::z_squared(std::span<const Rational>(data), mu0, pop_variance));
          },
          nb::arg("data"), nb::arg("mu0"), nb::arg("pop_variance"),
          "Exact rational z^2 = n (xbar - mu0)^2 / sigma2 for a known population variance, df1 = 1.");
    m.def("chi_squared_independence",
          [](const Matrix& table) { return unwrap(nimblecas::chi_squared_independence(table)); },
          nb::arg("table"),
          "Exact chi-squared test of independence on an r x c contingency table of counts, "
          "df1 = (r - 1)(c - 1).");
    m.def("exceeds",
          [](const Rational& statistic, const Rational& critical) {
              return unwrap(nimblecas::exceeds(statistic, critical));
          },
          nb::arg("statistic"), nb::arg("critical"),
          "Exact decision rule: statistic > critical by cross-multiplied rational comparison "
          "(pass the SQUARE of the critical value for a t^2 / z^2 statistic).");
    m.def("variance_ratio_f",
          [](const std::vector<Rational>& x, const std::vector<Rational>& y) {
              return unwrap(nimblecas::variance_ratio_f(std::span<const Rational>(x),
                                                        std::span<const Rational>(y)));
          },
          nb::arg("x"), nb::arg("y"),
          "Exact variance-ratio F = s_x^2 / s_y^2, df1 = n_x - 1, df2 = n_y - 1.");
    m.def("one_way_anova_f",
          [](const std::vector<std::vector<Rational>>& groups) {
              std::vector<std::span<const Rational>> spans;
              spans.reserve(groups.size());
              for (const auto& g : groups) {
                  spans.emplace_back(g);
              }
              return unwrap(nimblecas::one_way_anova_f(spans));
          },
          nb::arg("groups"),
          "Exact one-way ANOVA F over a list of group samples, df1 = k - 1, df2 = N - k.");

    // Point estimates from data (exact rational MLE).
    m.def("bernoulli_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::bernoulli_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Bernoulli MLE p-hat = sample mean (exact rational).");
    m.def("poisson_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::poisson_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Poisson MLE lambda-hat = sample mean (exact rational).");
    m.def("exponential_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::exponential_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Exponential MLE lambda-hat = 1 / sample mean (exact rational).");

    m.def("normal_mean_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::normal_mean_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Normal-mean MLE mu-hat = sample mean (exact rational).");
    m.def("geometric_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::geometric_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Geometric MLE p-hat = 1 / sample mean (exact rational).");
    m.def("normal_variance_mle",
          [](const std::vector<Rational>& data) {
              return unwrap(nimblecas::normal_variance_mle(std::span<const Rational>(data)));
          },
          nb::arg("data"), "Normal-variance MLE = the population variance (divide by n).");

    // Exact rational Wald / score statistics (closed forms are rational).
    m.def("bernoulli_wald_statistic",
          [](const std::vector<Rational>& data, const Rational& p0) {
              return unwrap(
                  nimblecas::bernoulli_wald_statistic(std::span<const Rational>(data), p0));
          },
          nb::arg("data"), nb::arg("p0"),
          "Bernoulli Wald W = n (p-hat - p0)^2 / (p-hat (1 - p-hat)) (exact rational).");
    m.def("bernoulli_score_statistic",
          [](const std::vector<Rational>& data, const Rational& p0) {
              return unwrap(
                  nimblecas::bernoulli_score_statistic(std::span<const Rational>(data), p0));
          },
          nb::arg("data"), nb::arg("p0"),
          "Bernoulli score (Rao) S = n (xbar - p0)^2 / (p0 (1 - p0)) (exact rational).");
    m.def("poisson_wald_statistic",
          [](const std::vector<Rational>& data, const Rational& lambda0) {
              return unwrap(
                  nimblecas::poisson_wald_statistic(std::span<const Rational>(data), lambda0));
          },
          nb::arg("data"), nb::arg("lambda0"),
          "Poisson Wald W = n (lambda-hat - lambda0)^2 / lambda-hat (exact rational).");
    m.def("poisson_score_statistic",
          [](const std::vector<Rational>& data, const Rational& lambda0) {
              return unwrap(
                  nimblecas::poisson_score_statistic(std::span<const Rational>(data), lambda0));
          },
          nb::arg("data"), nb::arg("lambda0"),
          "Poisson score (Rao) S = n (xbar - lambda0)^2 / lambda0 (exact rational).");

    // Symbolic MLE models (log-likelihood, score, estimator, Fisher information as Expr).
    m.def("bernoulli_mle_model", [] { return unwrap(nimblecas::bernoulli_mle_model()); },
          "The symbolic Bernoulli MLE model (MleModel).");
    m.def("poisson_mle_model", [] { return unwrap(nimblecas::poisson_mle_model()); },
          "The symbolic Poisson MLE model (MleModel).");
    m.def("exponential_mle_model", [] { return unwrap(nimblecas::exponential_mle_model()); },
          "The symbolic Exponential MLE model (MleModel).");
    m.def("normal_mean_mle_model", [] { return unwrap(nimblecas::normal_mean_mle_model()); },
          "The symbolic Normal-mean (known variance) MLE model (MleModel).");
    m.def("geometric_mle_model", [] { return unwrap(nimblecas::geometric_mle_model()); },
          "The symbolic Geometric MLE model (MleModel).");
    m.def(
        "log_likelihood_ratio",
        [](const MleModel& model, const Expr& theta_hat, const Expr& theta0) {
            return unwrap(nimblecas::log_likelihood_ratio(model, theta_hat, theta0));
        },
        nb::arg("model"), nb::arg("theta_hat"), nb::arg("theta0"),
        "The likelihood-ratio statistic G^2 = 2( ell(theta-hat) - ell(theta0) ), an exact "
        "symbolic Expr (transcendental in general).");

    // =======================================================================
    // QR decomposition & real Schur form (qrschur).
    // =======================================================================
    m.def("exact_orthogonal_qr",
          [](const Matrix& a) { return unwrap(nimblecas::exact_orthogonal_qr(a)); }, nb::arg("a"),
          "Exact Gram-Schmidt A = Q*R over Q: Q's columns are mutually orthogonal (not "
          "orthonormal), R upper-triangular with unit diagonal, Q*R == A exactly. Raises "
          "domain_error on a rank-deficient A.");
    m.def("numeric_qr",
          [](const std::vector<double>& a, std::size_t rows, std::size_t cols) {
              return unwrap(nimblecas::numeric_qr(std::span<const double>(a), rows, cols));
          },
          nb::arg("a"), nb::arg("rows"), nb::arg("cols"),
          "Numeric Householder QR of the row-major rows x cols matrix `a`: orthonormal Q (m x m) "
          "and upper-triangular R (m x n) with Q*R ~ A (NumericQr).");
    m.def("real_schur",
          [](const std::vector<double>& a, std::size_t n, double tol, std::size_t max_iter) {
              return unwrap(nimblecas::real_schur(std::span<const double>(a), n, tol, max_iter));
          },
          nb::arg("a"), nb::arg("n"), nb::arg("tol") = 1e-12,
          nb::arg("max_iter") = static_cast<std::size_t>(1000),
          "Numeric real Schur form A = Q*T*Q^T of the row-major n x n matrix `a` (NumericSchur); "
          "raises not_implemented on non-convergence.");
    m.def("schur_eigenvalues",
          [](const NumericSchur& s, double tol) {
              return unwrap(nimblecas::schur_eigenvalues(s, tol));
          },
          nb::arg("s"), nb::arg("tol") = 1e-9,
          "Eigenvalues read off the 1x1 / 2x2 diagonal blocks of a real Schur form (list of "
          "complex).");
    m.def("qr_residual",
          [](const NumericQr& d, const std::vector<double>& a) {
              return unwrap(nimblecas::qr_residual(d, std::span<const double>(a)));
          },
          nb::arg("d"), nb::arg("a"), "Frobenius residual ||Q*R - A||_F of a NumericQr.");
    m.def("schur_residual",
          [](const NumericSchur& s, const std::vector<double>& a) {
              return unwrap(nimblecas::schur_residual(s, std::span<const double>(a)));
          },
          nb::arg("s"), nb::arg("a"), "Frobenius residual ||Q*T*Q^T - A||_F of a NumericSchur.");
    m.def("orthonormality_defect",
          [](const std::vector<double>& q, std::size_t rows, std::size_t cols) {
              return unwrap(
                  nimblecas::orthonormality_defect(std::span<const double>(q), rows, cols));
          },
          nb::arg("q"), nb::arg("rows"), nb::arg("cols"),
          "Frobenius defect ||Q^T Q - I||_F of a row-major rows x cols matrix Q.");

    // =======================================================================
    // Numeric eigenvalues of a complex matrix (cheigen).
    // =======================================================================
    m.def("hermitian_eigenvalues",
          [](const ComplexMatrix& mat, double tol, std::size_t max_iter) {
              return unwrap(nimblecas::hermitian_eigenvalues(mat, tol, max_iter));
          },
          nb::arg("m"), nb::arg("tol") = 1e-12,
          nb::arg("max_iter") = static_cast<std::size_t>(1000),
          "All (numeric) eigenvalues of a Hermitian ComplexMatrix, returned REAL and ascending "
          "(list of float). A non-Hermitian matrix is rejected with a ValueError.");
    m.def("complex_eigenvalues",
          [](const ComplexMatrix& mat, double tol, std::size_t max_iter) {
              return unwrap(nimblecas::eigenvalues(mat, tol, max_iter));
          },
          nb::arg("m"), nb::arg("tol") = 1e-12,
          nb::arg("max_iter") = static_cast<std::size_t>(1000),
          "All (numeric) eigenvalues of a general ComplexMatrix as a list of complex (cheigen's "
          "eigenvalues). Raises not_implemented for a spectrum closed under conjugation (e.g. a "
          "real matrix with complex-conjugate eigenvalues — use eigenvalues_qr there).");

    // =======================================================================
    // Frobenius / rational canonical form over Q (frobenius). Exact: never needs
    // eigenvalues, stays entirely inside Q[x] via the Smith normal form of x*I - A.
    // =======================================================================
    m.def("invariant_factors",
          [](const Matrix& a) { return unwrap(nimblecas::invariant_factors(a)); }, nb::arg("a"),
          "The invariant factors f_1 | f_2 | ... | f_k of a square Matrix A: a list of monic "
          "RationalPoly, each dividing the next, whose product is the characteristic polynomial "
          "and whose last is the minimal polynomial. Exact over Q. Raises on a non-square A.");
    m.def("minimal_polynomial",
          [](const Matrix& a) { return unwrap(nimblecas::minimal_polynomial(a)); }, nb::arg("a"),
          "The minimal polynomial of A (its largest invariant factor) as a monic RationalPoly; "
          "the 0x0 matrix yields the constant 1. Raises on a non-square A.");
    m.def("companion_matrix",
          [](const RationalPoly& p) { return unwrap(nimblecas::companion_matrix(p)); },
          nb::arg("p"),
          "The companion Matrix C(p) of a degree>=1 polynomial (normalised monic first); its "
          "characteristic polynomial is the monic form of p. Raises on a zero/constant p.");
    m.def("rational_canonical_form",
          [](const Matrix& a) { return unwrap(nimblecas::rational_canonical_form(a)); },
          nb::arg("a"),
          "The rational canonical (Frobenius) form diag(C(f_1), ..., C(f_k)) of A, exact over Q. "
          "The transforming P with RCF(A) = P^{-1} A P is NOT computed. Raises on a non-square A.");

    // =======================================================================
    // Dynamical-systems stability (dynamics): exact phase-portrait and linear-
    // stability classification of dx/dt = A x, decided over Q with no root finding.
    // =======================================================================
    nb::enum_<PhaseType>(m, "PhaseType", "The qualitative 2x2 phase-portrait type.")
        .value("saddle", PhaseType::saddle)
        .value("node", PhaseType::node)
        .value("spiral", PhaseType::spiral)
        .value("center", PhaseType::center)
        .value("star", PhaseType::star)
        .value("degenerate_node", PhaseType::degenerate_node)
        .value("non_isolated", PhaseType::non_isolated);
    nb::enum_<Stability>(m, "Stability", "The stability verdict attached to a classification.")
        .value("stable", Stability::stable)
        .value("unstable", Stability::unstable)
        .value("neutrally_stable", Stability::neutrally_stable)
        .value("marginal", Stability::marginal);
    nb::enum_<LinearStability>(m, "LinearStability",
                              "The coarse nD Routh-Hurwitz linear-stability class.")
        .value("sink", LinearStability::sink)
        .value("source", LinearStability::source)
        .value("saddle", LinearStability::saddle)
        .value("borderline", LinearStability::borderline);

    // PhasePortrait: the full exact 2x2 trace-determinant verdict. Rational eigenvalue fields
    // are present only when exact (delta / -delta a perfect rational square), else None.
    nb::class_<PhasePortrait>(m, "PhasePortrait")
        .def_ro("type", &PhasePortrait::type)
        .def_ro("stability", &PhasePortrait::stability)
        .def_ro("trace", &PhasePortrait::trace, "T = tr(A).")
        .def_ro("determinant", &PhasePortrait::determinant, "D = det(A).")
        .def_ro("discriminant", &PhasePortrait::discriminant, "delta = T^2 - 4 D.")
        .def_ro("complex_eigenvalues", &PhasePortrait::complex_eigenvalues, "delta < 0.")
        .def_ro("repeated_eigenvalue", &PhasePortrait::repeated_eigenvalue, "delta == 0.")
        .def_ro("eigenvalues_rational", &PhasePortrait::eigenvalues_rational,
                "True when the exact rational eigenvalue fields below are complete.")
        .def_ro("lambda1", &PhasePortrait::lambda1,
                "Real case (T - sqrt(delta))/2 as a Rational, else None.")
        .def_ro("lambda2", &PhasePortrait::lambda2,
                "Real case (T + sqrt(delta))/2 as a Rational, else None.")
        .def_ro("real_part", &PhasePortrait::real_part,
                "Complex case T/2 as a Rational (always exact when complex), else None.")
        .def_ro("imag_part", &PhasePortrait::imag_part,
                "Complex case sqrt(-delta)/2 as a Rational when rational, else None.")
        .def_ro("description", &PhasePortrait::description)
        .def("__repr__",
             [](const PhasePortrait& p) { return "PhasePortrait(" + p.description + ")"; });

    // StabilityClassification: the coarse nD verdict from the Routh-Hurwitz sign-change count.
    nb::class_<StabilityClassification>(m, "StabilityClassification")
        .def_ro("dimension", &StabilityClassification::dimension)
        .def_ro("verdict", &StabilityClassification::verdict)
        .def_ro("rhp_count", &StabilityClassification::rhp_count,
                "# eigenvalues with Re > 0 when known, else -1 (borderline).")
        .def_ro("asymptotically_stable", &StabilityClassification::asymptotically_stable);

    m.def("is_perfect_square", &nimblecas::is_perfect_square, nb::arg("r"),
          "Whether the Rational r is the exact square of a rational (exact over Q; a negative r "
          "is never a real square).");
    m.def("classify_phase_portrait",
          [](const Matrix& a) { return unwrap(nimblecas::classify_phase_portrait(a)); },
          nb::arg("a"),
          "The exact 2x2 trace-determinant phase-portrait verdict (PhasePortrait). Requires a "
          "2x2 Matrix, else raises.");
    m.def("classify_linear_stability",
          [](const Matrix& a) { return unwrap(nimblecas::classify_linear_stability(a)); },
          nb::arg("a"),
          "The coarse nD Routh-Hurwitz stability class (StabilityClassification). Requires a "
          "square Matrix with n >= 1.");
    m.def("fixed_point_affine",
          [](const Matrix& a, const Matrix& b) {
              return unwrap(nimblecas::fixed_point_affine(a, b));
          },
          nb::arg("a"), nb::arg("b"),
          "The equilibrium of x |-> A x + b: the exact solution of (I - A) x = b over Q (A "
          "square n x n, b an n x 1 column). Raises on a non-isolated equilibrium.");
    m.def("is_asymptotically_stable",
          [](const Matrix& a) { return unwrap(nimblecas::is_asymptotically_stable(a)); },
          nb::arg("a"),
          "Whether dx/dt = A x is asymptotically stable, decided exactly by the Routh-Hurwitz "
          "criterion on the characteristic polynomial (no root finding). Requires n >= 1.");
    m.def("classify_equilibrium",
          [](const Matrix& a) { return unwrap(nimblecas::classify_equilibrium(a)); }, nb::arg("a"),
          "A human-readable classification string of the origin of dx/dt = A x.");

    // =======================================================================
    // Algebraic number fields Q(alpha) = Q[x]/(m) (algnum): exact arithmetic in
    // a simple algebraic extension. NumberField is the factory; AlgebraicNumber
    // is a field element carried as a canonical RationalPoly residue (degree < d).
    // NumberField is registered before AlgebraicNumber, and both before jordan's
    // AlgebraicJordan, which returns matrices of AlgebraicNumber.
    // =======================================================================
    nb::class_<NumberField>(m, "NumberField")
        .def_static(
            "create",
            [](const RationalPoly& minimal) { return unwrap(NumberField::create(minimal)); },
            nb::arg("minimal"),
            "Build Q[x]/(m) from a minimal polynomial m of degree >= 1 (normalised to monic); a "
            "reducible, constant, or zero m raises (irreducibility is verified over Q).")
        .def("degree", &NumberField::degree, "The extension degree d = deg m = [Q(alpha):Q].")
        .def("modulus", [](const NumberField& f) { return f.modulus(); },
             "The monic irreducible minimal polynomial m as a RationalPoly.")
        .def("is_same", &NumberField::is_same, nb::arg("other"),
             "Whether two fields are equal (identical minimal polynomials).")
        .def("zero", &NumberField::zero, "The field element 0.")
        .def("one", &NumberField::one, "The field element 1.")
        .def("from_rational", &NumberField::from_rational, nb::arg("c"),
             "The constant element c in Q(alpha).")
        .def("generator", [](const NumberField& f) { return unwrap(f.generator()); },
             "The generator alpha = x mod m.")
        .def("from_poly",
             [](const NumberField& f, const RationalPoly& p) { return unwrap(f.from_poly(p)); },
             nb::arg("p"), "The class of an arbitrary polynomial p in Q[x], i.e. p reduced mod m.")
        .def("to_string", [](const NumberField& f) { return f.to_string(); })
        .def("to_string", [](const NumberField& f, const std::string& var) {
            return f.to_string(var);
        }, nb::arg("var"))
        .def("__repr__", [](const NumberField& f) { return f.to_string(); })
        .def("__eq__",
             [](const NumberField& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<NumberField>(other)) {
                     return nb::cast(a.is_same(nb::cast<NumberField>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__", [](const NumberField& a, nb::handle other) -> nb::object {
            if (nb::isinstance<NumberField>(other)) {
                return nb::cast(!a.is_same(nb::cast<NumberField>(other)));
            }
            return nb::borrow(Py_NotImplemented);
        });

    nb::class_<AlgebraicNumber>(m, "AlgebraicNumber")
        .def("field", [](const AlgebraicNumber& a) { return a.field(); },
             "The NumberField this element belongs to.")
        .def("value", [](const AlgebraicNumber& a) { return a.value(); },
             "The canonical residue: a RationalPoly of degree < d in the basis 1, alpha, ..., "
             "alpha^(d-1).")
        .def("is_zero", &AlgebraicNumber::is_zero)
        .def("is_one", &AlgebraicNumber::is_one)
        .def("is_equal", &AlgebraicNumber::is_equal, nb::arg("other"),
             "Equality: same field AND same residue.")
        .def("add", [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.add(b)); },
             nb::arg("other"))
        .def("subtract",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.subtract(b)); },
             nb::arg("other"))
        .def("negate", [](const AlgebraicNumber& a) { return unwrap(a.negate()); })
        .def("multiply",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.multiply(b)); },
             nb::arg("other"))
        .def("inverse", [](const AlgebraicNumber& a) { return unwrap(a.inverse()); },
             "The multiplicative inverse (raises division_by_zero on the zero element).")
        .def("divide",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.divide(b)); },
             nb::arg("other"))
        .def("pow", [](const AlgebraicNumber& a, std::int64_t exponent) { return unwrap(a.pow(exponent)); },
             nb::arg("exponent"),
             "Non-negative integer power by repeated squaring (a negative exponent raises).")
        .def("norm", [](const AlgebraicNumber& a) { return unwrap(a.norm()); },
             "The field norm N(a), an exact Rational (the determinant of multiplication-by-a).")
        .def("trace", [](const AlgebraicNumber& a) { return unwrap(a.trace()); },
             "The field trace Tr(a), an exact Rational (the trace of multiplication-by-a).")
        .def("to_string", [](const AlgebraicNumber& a) { return a.to_string(); })
        .def("to_string", [](const AlgebraicNumber& a, const std::string& var) {
            return a.to_string(var);
        }, nb::arg("var"))
        .def("__repr__", [](const AlgebraicNumber& a) { return a.to_string(); })
        .def("__add__",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.add(b)); })
        .def("__sub__",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.subtract(b)); })
        .def("__mul__",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.multiply(b)); })
        .def("__truediv__",
             [](const AlgebraicNumber& a, const AlgebraicNumber& b) { return unwrap(a.divide(b)); })
        .def("__neg__", [](const AlgebraicNumber& a) { return unwrap(a.negate()); })
        .def("__eq__",
             [](const AlgebraicNumber& a, nb::handle other) -> nb::object {
                 if (nb::isinstance<AlgebraicNumber>(other)) {
                     return nb::cast(a.is_equal(nb::cast<AlgebraicNumber>(other)));
                 }
                 return nb::borrow(Py_NotImplemented);
             })
        .def("__ne__", [](const AlgebraicNumber& a, nb::handle other) -> nb::object {
            if (nb::isinstance<AlgebraicNumber>(other)) {
                return nb::cast(!a.is_equal(nb::cast<AlgebraicNumber>(other)));
            }
            return nb::borrow(Py_NotImplemented);
        });

    // =======================================================================
    // Jordan canonical form with transform (jordan). TIER 1 rational_jordan_form
    // is exact over Q (char poly splits over Q); TIER 2 jordan_form works over a
    // single quadratic extension Q(alpha), returning AlgebraicNumber matrices.
    // =======================================================================
    nb::class_<RationalJordan>(m, "RationalJordan")
        .def_ro("jordan", &RationalJordan::jordan,
                "The block-diagonal Jordan matrix J (exact Rational Matrix).")
        .def_ro("transform", &RationalJordan::transform,
                "The invertible P (Matrix) with A*P == P*J exactly.");
    m.def("rational_jordan_form",
          [](const Matrix& a) { return unwrap(nimblecas::rational_jordan_form(a)); }, nb::arg("a"),
          "TIER 1 Jordan form over Q (RationalJordan {jordan, transform}), valid when the "
          "characteristic polynomial splits over Q. A*P == P*J and P invertible are verified "
          "exactly. Raises domain_error on a non-square A or a char poly that does not split "
          "over Q (use jordan_form there).");

    nb::class_<AlgebraicJordan>(m, "AlgebraicJordan")
        .def_ro("field", &AlgebraicJordan::field,
                "The simple quadratic extension Q(alpha) the eigenvalues live in (NumberField).")
        .def_ro("jordan", &AlgebraicJordan::jordan,
                "The Jordan matrix J as list[list[AlgebraicNumber]] over Q(alpha).")
        .def_ro("transform", &AlgebraicJordan::transform,
                "The transform P as list[list[AlgebraicNumber]] with A*P == P*J over Q(alpha).");
    m.def("jordan_form",
          [](const Matrix& a) { return unwrap(nimblecas::jordan_form(a)); }, nb::arg("a"),
          "TIER 2 Jordan form over a single quadratic extension Q(alpha) (AlgebraicJordan "
          "{field, jordan, transform}), valid when the char poly's only non-linear irreducible "
          "factor is one quadratic. Raises domain_error when it splits over Q (use "
          "rational_jordan_form) and not_implemented for degree>=3 or multiple distinct "
          "quadratic factors.");

    // =======================================================================
    // Probability-distribution catalog (probdist) — exact symbolic MGF/PGF,
    // mean, variance, and moments/cumulants as Expr trees. A submodule so the
    // distribution constructors and moment extractors group cleanly.
    // =======================================================================
    nb::module_ probdist = m.def_submodule(
        "probdist",
        "Exact symbolic probability-distribution catalog: each constructor returns a DistInfo "
        "carrying the moment generating function mgf (or None), the probability generating "
        "function pgf (or None), and the mean / variance as Expr trees in the parameter symbols.");
    nb::class_<DistInfo>(probdist, "DistInfo")
        .def_ro("mgf", &DistInfo::mgf, "Moment generating function M_X(t) as an Expr, or None.")
        .def_ro("pgf", &DistInfo::pgf, "Probability generating function G_X(z) as an Expr, or None.")
        .def_ro("mean", &DistInfo::mean, "E[X] as an Expr.")
        .def_ro("variance", &DistInfo::variance, "Var(X) as an Expr.");
    probdist.def("bernoulli", &nimblecas::bernoulli, nb::arg("p"));
    probdist.def("binomial", &nimblecas::binomial, nb::arg("n"), nb::arg("p"));
    probdist.def("poisson", &nimblecas::poisson, nb::arg("lambda"));
    probdist.def("geometric", &nimblecas::geometric, nb::arg("p"));
    probdist.def("exponential", &nimblecas::exponential, nb::arg("lambda"));
    probdist.def("normal", &nimblecas::normal, nb::arg("mu"), nb::arg("sigma2"),
                 "Normal(mu, sigma2) where sigma2 is the variance.");
    probdist.def("gamma", &nimblecas::gamma, nb::arg("alpha"), nb::arg("theta"),
                 "Gamma(alpha, theta) with shape alpha and scale theta.");
    probdist.def("discrete_uniform", &nimblecas::discrete_uniform, nb::arg("a"), nb::arg("b"));
    probdist.def("negative_binomial", &nimblecas::negative_binomial, nb::arg("r"), nb::arg("p"));
    probdist.def("hypergeometric", &nimblecas::hypergeometric, nb::arg("N"), nb::arg("K"),
                 nb::arg("n"));
    probdist.def("continuous_uniform", &nimblecas::continuous_uniform, nb::arg("a"), nb::arg("b"));
    probdist.def("chi_squared", &nimblecas::chi_squared, nb::arg("k"));
    probdist.def("student_t", &nimblecas::student_t, nb::arg("nu"));
    probdist.def("beta", &nimblecas::beta, nb::arg("alpha"), nb::arg("beta"));
    probdist.def("weibull", &nimblecas::weibull, nb::arg("k"), nb::arg("lambda"));
    probdist.def("pareto", &nimblecas::pareto, nb::arg("xm"), nb::arg("alpha"));
    probdist.def("lognormal", &nimblecas::lognormal, nb::arg("mu"), nb::arg("sigma2"));
    probdist.def("raw_moment",
                 [](const Expr& mgf, std::size_t k) {
                     return unwrap(nimblecas::raw_moment(mgf, k));
                 },
                 nb::arg("mgf"), nb::arg("k"),
                 "k-th raw moment E[X^k] = [d^k/dt^k M_X(t)]_{t=0} (exact; may be unsimplified).");
    probdist.def("cumulant",
                 [](const Expr& mgf, std::size_t k) {
                     return unwrap(nimblecas::cumulant(mgf, k));
                 },
                 nb::arg("mgf"), nb::arg("k"),
                 "k-th cumulant [d^k/dt^k log M_X(t)]_{t=0} (kappa_1 = mean, kappa_2 = variance).");
    probdist.def("characteristic_function", &nimblecas::characteristic_function, nb::arg("mgf"),
                 "phi_X(t) = M_X(i t) by the formal substitution t -> i*t.");
    probdist.def("laplace_stieltjes", &nimblecas::laplace_stieltjes, nb::arg("mgf"),
                 "LST_X(s) = M_X(-s) by the formal substitution t -> -s.");
    probdist.def("markov_bound", &nimblecas::markov_bound, nb::arg("mean"), nb::arg("alpha"),
                 "Markov tail bound E[X]/alpha (RHS of P(X >= alpha) <= E[X]/alpha).");
    probdist.def("chebyshev_bound", &nimblecas::chebyshev_bound, nb::arg("variance"), nb::arg("k"),
                 "Chebyshev tail bound sigma^2 / k^2.");
    probdist.def("factorial_moment",
                 [](const Expr& pgf, std::size_t k) {
                     return unwrap(nimblecas::factorial_moment(pgf, k));
                 },
                 nb::arg("pgf"), nb::arg("k"),
                 "k-th factorial moment E[X(X-1)...(X-k+1)] = [d^k/dz^k G_X(z)]_{z=1} from a PGF "
                 "(exact; may be unsimplified).");
    probdist.def("cantelli_bound", &nimblecas::cantelli_bound, nb::arg("variance"), nb::arg("k"),
                 "Cantelli one-sided tail bound sigma^2 / (sigma^2 + k^2).");
    probdist.def("chernoff_bound", &nimblecas::chernoff_bound, nb::arg("mgf"), nb::arg("alpha"),
                 "Chernoff bound e^{-t alpha} M_X(t), for every t > 0 (minimise over t externally).");
    probdist.def("hoeffding_bound", &nimblecas::hoeffding_bound, nb::arg("t"), nb::arg("widths"),
                 "Hoeffding bound exp(-2 t^2 / sum_i widths_i^2); widths are the ranges b_i - a_i.");
    probdist.def("bernstein_bound", &nimblecas::bernstein_bound, nb::arg("t"), nb::arg("variance"),
                 nb::arg("bound"),
                 "Bernstein bound exp(-t^2 / (2 (v + M t / 3))) with total variance v and |X_i| <= M.");
    probdist.def("mcdiarmid_bound", &nimblecas::mcdiarmid_bound, nb::arg("t"), nb::arg("diffs"),
                 "McDiarmid bounded-differences bound exp(-2 t^2 / sum_i diffs_i^2).");
    probdist.def("azuma_bound", &nimblecas::azuma_bound, nb::arg("t"), nb::arg("diffs"),
                 "Azuma-Hoeffding martingale bound exp(-t^2 / (2 sum_i diffs_i^2)).");

    // =======================================================================
    // GPU acceleration (gpu) — bound only when compiled with
    // -DNIMBLECAS_EXT_WITH_GPU and linked against nimblecas_gpu + CUDA runtime.
    // HAS_GPU tells Python whether the submodule is present.
    // =======================================================================
#ifdef NIMBLECAS_EXT_WITH_GPU
    m.attr("HAS_GPU") = true;
    nb::module_ gpu = m.def_submodule(
        "gpu", "CUDA-accelerated kernels (present only in a CUDA-enabled build; see HAS_GPU).");
    gpu.def("device_count", &nimblecas::gpu::device_count,
            "Number of CUDA-capable devices detected (0 when no GPU is present).");
    gpu.def("available", &nimblecas::gpu::available, "Whether at least one GPU is available.");
    gpu.def("poly_eval",
            [](const std::vector<double>& coeffs, const std::vector<double>& x) {
                return unwrap(nimblecas::gpu::poly_eval(std::span<const double>(coeffs),
                                                        std::span<const double>(x)));
            },
            nb::arg("coeffs"), nb::arg("x"),
            "Evaluate the polynomial `coeffs` (low degree first) at every point of `x` on the GPU.");
    gpu.def("edit_distance_batch",
            [](const std::vector<int>& a_flat, const std::vector<int>& a_off,
               const std::vector<int>& b_flat, const std::vector<int>& b_off) {
                return unwrap(nimblecas::gpu::edit_distance_batch(
                    std::span<const int>(a_flat), std::span<const int>(a_off),
                    std::span<const int>(b_flat), std::span<const int>(b_off)));
            },
            nb::arg("a_flat"), nb::arg("a_off"), nb::arg("b_flat"), nb::arg("b_off"),
            "Batched Levenshtein edit distance over flattened code-point arrays with prefix-offset "
            "arrays (one distance per pair).");
    gpu.def("bfs",
            [](const std::vector<int>& row_offsets, const std::vector<int>& col_indices,
               int source) {
                return unwrap(nimblecas::gpu::bfs(std::span<const int>(row_offsets),
                                                  std::span<const int>(col_indices), source));
            },
            nb::arg("row_offsets"), nb::arg("col_indices"), nb::arg("source"),
            "Level-synchronous single-source BFS over a CSR graph (hop distance, -1 unreachable).");
    gpu.def("nqueens_count",
            [](int n) { return unwrap(nimblecas::gpu::nqueens_count(n)); }, nb::arg("n"),
            "Count the solutions to the n-queens problem on the GPU.");
    gpu.def("qmc_poly_integrate",
            [](const std::vector<double>& coeffs, const std::vector<double>& points) {
                return unwrap(nimblecas::gpu::qmc_poly_integrate(std::span<const double>(coeffs),
                                                                 std::span<const double>(points)));
            },
            nb::arg("coeffs"), nb::arg("points"),
            "Equal-weight average (1/N) sum_i p(points_i) of the polynomial integrand on the GPU.");
    gpu.def("haar_dwt_batch",
            [](const std::vector<double>& data, int batch, int len) {
                return unwrap(nimblecas::gpu::haar_dwt_batch(std::span<const double>(data), batch,
                                                             len));
            },
            nb::arg("data"), nb::arg("batch"), nb::arg("len"),
            "One-level batched Haar DWT (orthonormal 1/sqrt(2)) over `batch` blocks of `len` "
            "(even) samples each, row-major.");
    gpu.def("batched_matmul",
            [](const std::vector<double>& a, const std::vector<double>& b, int batch, int m,
               int k, int n) {
                return unwrap(nimblecas::gpu::batched_matmul(std::span<const double>(a),
                                                             std::span<const double>(b), batch, m,
                                                             k, n));
            },
            nb::arg("a"), nb::arg("b"), nb::arg("batch"), nb::arg("m"), nb::arg("k"), nb::arg("n"),
            "Batched double dense matrix multiply on the GPU: `batch` independent products "
            "C_b = A_b (m x k) * B_b (k x n), all row-major and contiguously packed. Numeric.");
    gpu.def("fft_batch",
            [](const std::vector<double>& in, int batch, int n) {
                return unwrap(nimblecas::gpu::fft_batch(std::span<const double>(in), batch, n));
            },
            nb::arg("in"), nb::arg("batch"), nb::arg("n"),
            "Batched radix-2 FFT of `batch` complex signals of length n (a power of two, "
            "n <= 2048), each supplied as 2*n interleaved (real, imag) doubles contiguously "
            "packed; returns the batch*2*n interleaved (real, imag) spectra.");
#else
    m.attr("HAS_GPU") = false;
#endif
}
