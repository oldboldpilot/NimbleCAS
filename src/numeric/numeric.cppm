// NimbleCAS numeric module: real-root finders for polynomials (ROADMAP §7.14).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions).
//
// This is a NUMERIC module: polynomials are represented by their double-precision
// coefficients in ascending order (low degree first), passed as std::span<const
// double>. It is deliberately self-contained and does NOT depend on the exact
// nimblecas.polynomial module — §7.14 is about floating-point solvers.

export module nimblecas.numeric;

import std;
import nimblecas.core;

export namespace nimblecas::numeric {

// ---------------------------------------------------------------------------
// Internal evaluation helpers (Horner's scheme).
// ---------------------------------------------------------------------------

// Evaluates p(x) = c[0] + c[1] x + ... + c[n-1] x^(n-1) by Horner's scheme.
// An empty coefficient span denotes the zero polynomial, whose value is 0.
[[nodiscard]] auto eval(std::span<const double> c, double x) noexcept -> double {
    double acc = 0.0;
    for (double coeff : std::ranges::reverse_view(c)) {
        acc = std::fma(acc, x, coeff);
    }
    return acc;
}

// Evaluates p'(x) for the same ascending-order coefficient span. The derivative
// coefficients are d[k-1] = k * c[k], differentiated on the fly so no temporary
// buffer is allocated. The zero and constant polynomials both have derivative 0.
[[nodiscard]] auto eval_derivative(std::span<const double> c, double x) noexcept -> double {
    double acc = 0.0;
    // Walk k = n-1 .. 1, accumulating Horner on the derivative coefficients k*c[k].
    for (std::size_t k = c.size(); k-- > 1;) {
        acc = std::fma(acc, x, static_cast<double>(k) * c[k]);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Root finders. All return the located root, or a MathError:
//   * domain_error  — empty coefficients, or a bracket with no sign change.
//   * not_implemented — failed to converge within max_iter, or a (near-)zero
//                       derivative / secant denominator stalled the iteration.
// ---------------------------------------------------------------------------

// Newton-Raphson: x_{n+1} = x_n - p(x_n) / p'(x_n). Converges when |p(x)| <= tol
// or the step magnitude falls below tol.
[[nodiscard]] auto newton(std::span<const double> coeffs, double x0, double tol,
                          int max_iter) -> Result<double> {
    if (coeffs.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double x = x0;
    for (int iter = 0; iter < max_iter; ++iter) {
        const double fx = eval(coeffs, x);
        if (std::abs(fx) <= tol) {
            return x;
        }
        const double dfx = eval_derivative(coeffs, x);
        if (std::abs(dfx) <= tol) {
            return make_error<double>(MathError::not_implemented);  // flat derivative
        }
        const double step = fx / dfx;
        x -= step;
        if (std::abs(step) <= tol) {
            return x;
        }
    }
    return make_error<double>(MathError::not_implemented);  // exhausted iterations
}

// Bisection: requires a sign change on [a, b] (p(a)*p(b) <= 0), then halves the
// bracket until it is narrower than tol and returns the midpoint.
[[nodiscard]] auto bisection(std::span<const double> coeffs, double a, double b,
                             double tol) -> Result<double> {
    if (coeffs.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double lo = a;
    double hi = b;
    double flo = eval(coeffs, lo);
    const double fhi = eval(coeffs, hi);
    if (flo * fhi > 0.0) {
        return make_error<double>(MathError::domain_error);  // no bracketed root
    }
    // Roots exactly on an endpoint must be returned directly: with flo == 0 the sign
    // test flo*fmid < 0.0 can never fire, so the bracket would otherwise drift off the
    // true root toward the opposite bound. (fhi == 0 converges anyway, but is symmetric.)
    if (flo == 0.0) {
        return lo;
    }
    if (fhi == 0.0) {
        return hi;
    }
    while (std::abs(hi - lo) >= tol) {
        const double mid = std::midpoint(lo, hi);
        const double fmid = eval(coeffs, mid);
        if (fmid == 0.0) {
            return mid;  // exact root hit
        }
        if (flo * fmid < 0.0) {
            hi = mid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    return std::midpoint(lo, hi);
}

// Secant: derivative-free Newton variant using a finite-difference slope from the
// two most recent iterates. Same convergence/failure contract as newton.
[[nodiscard]] auto secant(std::span<const double> coeffs, double x0, double x1,
                          double tol, int max_iter) -> Result<double> {
    if (coeffs.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double xprev = x0;
    double xcurr = x1;
    double fprev = eval(coeffs, xprev);
    for (int iter = 0; iter < max_iter; ++iter) {
        const double fcurr = eval(coeffs, xcurr);
        if (std::abs(fcurr) <= tol) {
            return xcurr;
        }
        const double denom = fcurr - fprev;
        if (std::abs(denom) <= tol) {
            return make_error<double>(MathError::not_implemented);  // slope collapsed
        }
        const double step = fcurr * (xcurr - xprev) / denom;
        xprev = xcurr;
        fprev = fcurr;
        xcurr -= step;
        if (std::abs(step) <= tol) {
            return xcurr;
        }
    }
    return make_error<double>(MathError::not_implemented);  // exhausted iterations
}

}  // namespace nimblecas::numeric
