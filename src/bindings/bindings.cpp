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
//   Value types : Expr, Rational, RationalPoly, Matrix, Complex.
//   Small POD   : Root, TestStatistic, PeriodicCF, MleModel (read-only fields).
//   Symbolic    : free_of, substitute, simplify, differentiate, polynomial_gcd,
//                 square_free_factor, expand, integrate, integrate_definite.
//   Polynomial  : solve_poly, factor_over_Q, companion_eigenvalues, eigenvalues_qr.
//   Linear alg. : kronecker_product / _sum, direct_sum, hadamard_product, vec, unvec.
//   Number thy. : from_rational, convergents, reconstruct, quadratic_irrational_cf.
//   Statistics  : mean, variance, median, covariance, chi_squared_goodness_of_fit,
//                 variance_ratio_f, one_way_anova_f, bernoulli/poisson/exponential MLE
//                 (both the symbolic model and the exact point estimate from data).

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
import nimblecas.solve;
import nimblecas.factor;
import nimblecas.expand;
import nimblecas.symint;
import nimblecas.numeigen;
import nimblecas.stats;
import nimblecas.kronprod;
import nimblecas.contfrac;
import nimblecas.hyptest;

namespace nb = nanobind;
using nimblecas::Complex;
using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::MleModel;
using nimblecas::PeriodicCF;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::Root;
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

    // Symbolic MLE models (log-likelihood, score, estimator, Fisher information as Expr).
    m.def("bernoulli_mle_model", [] { return unwrap(nimblecas::bernoulli_mle_model()); },
          "The symbolic Bernoulli MLE model (MleModel).");
    m.def("poisson_mle_model", [] { return unwrap(nimblecas::poisson_mle_model()); },
          "The symbolic Poisson MLE model (MleModel).");
    m.def("exponential_mle_model", [] { return unwrap(nimblecas::exponential_mle_model()); },
          "The symbolic Exponential MLE model (MleModel).");
}
