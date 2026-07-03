// NimbleCAS rational-function integration capstone (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// integrate_rational assembles the complete indefinite integral of a rational function
// A(x)/B(x) over Q by running the two halves of the standard decomposition in sequence:
//
//   1. Hermite reduction (nimblecas.ratint) splits off the elementary rational part g and
//      leaves a proper integrand with a square-free denominator.
//   2. Rothstein-Trager (nimblecas.rothstein) integrates that square-free remainder into a
//      sum of logarithms with constant (residue) multipliers.
//
// The result is  int A/B dx = rational_num/rational_den + sum_i c_i * log(argument_i).
// When a residue is irrational or complex the logarithmic step returns
// MathError::not_implemented and it propagates here: the integral is not expressible in
// this rational-plus-rational-logarithm form without an algebraic extension field.

export module nimblecas.integrate;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.ratint;
import nimblecas.rothstein;

export namespace nimblecas {

// The complete integral:
//     int numerator/denominator dx
//         == rational_num/rational_den + sum over log_terms of coefficient * log(argument).
struct RationalIntegral {
    RationalPoly rational_num;         // g numerator (the elementary rational part)
    RationalPoly rational_den;         // g denominator (never zero; constant 1 when g == 0)
    std::vector<LogTerm> log_terms;    // the logarithmic part; empty when purely rational
};

// Integrate numerator/denominator over Q(x). Fails with division_by_zero (a zero
// denominator), not_implemented (a non-rational residue in the logarithmic part), or
// overflow (an int64 coefficient limit).
[[nodiscard]] auto integrate_rational(const RationalPoly& numerator,
                                      const RationalPoly& denominator)
    -> Result<RationalIntegral>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto integrate_rational(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<RationalIntegral> {
    // Rational part + square-free logarithmic integrand.
    auto hr = hermite_reduce(numerator, denominator);
    if (!hr) {
        return make_error<RationalIntegral>(hr.error());
    }
    RationalIntegral out;
    out.rational_num = std::move(hr->rational_num);
    out.rational_den = std::move(hr->rational_den);
    // Integrate the leftover square-free integrand into logarithms, if any remains.
    if (!hr->integrand_num.is_zero()) {
        auto lp = log_part(hr->integrand_num, hr->integrand_den);
        if (!lp) {
            return make_error<RationalIntegral>(lp.error());
        }
        out.log_terms = std::move(lp->terms);
    }
    return out;
}

}  // namespace nimblecas
