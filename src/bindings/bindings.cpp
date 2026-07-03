// NimbleCAS Python bindings via nanobind (ROADMAP 10).
// @author Olumuyiwa Oluwasanmi
//
// This is a non-module translation unit: it #includes the nanobind headers (which
// pull in the standard library classically) and `import`s the NimbleCAS C++23
// modules. The std::string / std::vector types from the headers and from the
// imported modules are the same entities, so they interoperate directly.
//
// The C++ core is exception-free (Rule 32, std::expected). At the Python boundary a
// MathError is translated into a Python exception, which is the idiomatic Python
// contract.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.polyexpr;

namespace nb = nanobind;
using nimblecas::Expr;
using nimblecas::MathError;

namespace {

// Unwraps a Result<Expr>, raising a Python exception carrying the MathError text.
[[nodiscard]] auto unwrap(nimblecas::Result<Expr> result) -> Expr {
    if (!result.has_value()) {
        throw nb::value_error(nimblecas::to_string_view(result.error()).data());
    }
    return std::move(result).value();
}

}  // namespace

NB_MODULE(nimblecas_ext, m) {
    m.doc() = "NimbleCAS Python bindings (symbolic core: Expr, FreeOf, Substitute)";

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
        [](const Expr& u, const std::string& var)
            -> std::vector<std::pair<Expr, std::int64_t>> {
            auto factors = nimblecas::square_free_factor(u, var);
            if (!factors.has_value()) {
                throw nb::value_error(nimblecas::to_string_view(factors.error()).data());
            }
            return std::move(factors).value();
        },
        nb::arg("u"), nb::arg("var"),
        "Square-free factorization: list of (factor, multiplicity) for a polynomial in var.");
}
